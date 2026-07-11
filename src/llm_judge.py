"""Helpers for shaping evaluator definitions for the calibrate CLI.

Handles `{{variable}}` substitution and rubric assembly. The actual LLM-as-judge
HTTP call lives in the calibrate CLI — this module only renders prompts and
builds the config payload the CLI expects.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render_template(template: str, variables: Dict[str, Any]) -> str:
    """Replace `{{name}}` in `template` with values from `variables`. Missing vars render as ''."""

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        value = variables.get(name, "")
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _VARIABLE_PATTERN.sub(_sub, template)


def _format_scale_rubric(output_config: Optional[Dict[str, Any]]) -> str:
    """If any scale entry has a `description`, return a multi-line rubric block the judge
    can read to understand what each value means. Empty string when no descriptions present.

    Format:
        Rubric:
          1 (Poor): Garbled or missing content.
          2 (Weak): Significant issues.
          ...
    """
    scale = (output_config or {}).get("scale")
    if not isinstance(scale, list):
        return ""
    lines = []
    for entry in scale:
        desc = entry.get("description")
        if not desc:
            continue
        value = entry.get("value")
        name = entry.get("name")
        label = f"{value}" + (f" ({name})" if name else "")
        lines.append(f"  {label}: {desc}")
    if not lines:
        return ""
    return "\n\nRubric:\n" + "\n".join(lines)


_DEFAULT_BINARY_OUTPUT_CONFIG: Dict[str, Any] = {
    "scale": [
        {"value": True, "name": "Correct"},
        {"value": False, "name": "Wrong"},
    ]
}


def default_output_config(output_type: Optional[str]) -> Optional[Dict[str, Any]]:
    """Sensible default `output_config` for versions that don't carry their
    own rubric. Binary → Correct/Wrong scale (mirrors the
    `evaluator_value_name` fallback). Rating returns None because no meaningful
    default scale exists without bounds — `evaluator_value_name` falls back to
    `str(value)` there, which the FE can replicate. Returns a deep-ish copy so
    callers can mutate it without leaking back into the constant."""
    if output_type == "binary":
        return {"scale": [dict(e) for e in _DEFAULT_BINARY_OUTPUT_CONFIG["scale"]]}
    return None


def evaluator_value_name(
    value: Any,
    output_type: Optional[str],
    output_config: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Map an evaluator-run scalar to its display name. Prefers an explicit
    `name` from `output_config.scale`; falls back to `Correct`/`Wrong` for
    binary true/false, and stringified score for rating."""
    if value is None:
        return None
    scale = (output_config or {}).get("scale")
    if isinstance(scale, list):
        for entry in scale:
            if entry.get("value") == value and entry.get("name"):
                return entry["name"]
    if output_type == "binary":
        if value is True:
            return "Correct"
        if value is False:
            return "Wrong"
    if output_type == "rating" and isinstance(value, (int, float)):
        return str(value)
    return None


def _scale_bounds(output_config: Optional[Dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
    """For rating evaluators, derive scale_min/scale_max from output_config.scale.

    Calibrate constrains the judge to integers in [scale_min, scale_max] (Pydantic Literal).
    """
    scale = (output_config or {}).get("scale")
    if not isinstance(scale, list) or not scale:
        return (None, None)
    numeric_values = [e.get("value") for e in scale if isinstance(e.get("value"), (int, float))]
    if not numeric_values:
        return (None, None)
    return (min(numeric_values), max(numeric_values))


def _calibrate_evaluator_def(
    ev: Dict[str, Any],
    rendered_prompt: str,
) -> Dict[str, Any]:
    """Shape a single evaluator dict into calibrate's expected contract."""
    out: Dict[str, Any] = {
        "name": ev.get("name"),
        "system_prompt": rendered_prompt,
        "judge_model": ev.get("judge_model"),
        "type": ev.get("output_type", "binary"),
    }
    if ev.get("uuid"):
        out["id"] = ev["uuid"]
    if out["type"] == "rating":
        scale_min, scale_max = _scale_bounds(ev.get("output_config"))
        if scale_min is not None and scale_max is not None:
            out["scale_min"] = scale_min
            out["scale_max"] = scale_max
    return out


def _render_with_rubric(
    ev: Dict[str, Any],
    extra_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """Substitute {{variable}} placeholders in the evaluator's system_prompt and append the
    per-level rubric block when scale entries have descriptions."""
    variables_spec = ev.get("variables") or []
    variable_values = dict(ev.get("variable_values") or {})
    if extra_vars:
        variable_values.update(extra_vars)
    render_vars: Dict[str, Any] = {}
    for spec in variables_spec:
        name = spec.get("name")
        if not name:
            continue
        render_vars[name] = variable_values.get(name, spec.get("default", ""))
    for k, v in variable_values.items():
        render_vars.setdefault(k, v)

    rendered = render_template(ev.get("system_prompt", ""), render_vars)
    rubric = _format_scale_rubric(ev.get("output_config"))
    if rubric:
        rendered = rendered.rstrip() + rubric
    return rendered


def build_evaluator_cli_payload(
    evaluators: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render each linked evaluator into the dict shape calibrate expects.

    Use for STT, TTS, and simulation runs — variables are pre-substituted into the
    system_prompt because those flows don't have a per-row arguments mechanism. For LLM
    tests/benchmarks, use `build_test_evaluators_payload` instead so each test case can
    pass its own `arguments`.
    """
    payload = []
    for ev in evaluators:
        rendered_prompt = _render_with_rubric(ev)
        payload.append(_calibrate_evaluator_def(ev, rendered_prompt))
    return payload


def refresh_evaluators_to_live(
    snapshot_evaluators: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Re-hydrate submit-time evaluator snapshots to each evaluator's CURRENT
    live version at run time.

    STT/TTS jobs snapshot a fully-hydrated evaluator into ``details.evaluators``
    at submit time. To make runs use the LIVE version (consistent with LLM tests
    and simulations, which resolve live), we re-read each evaluator's current
    ``live_version_id`` when the job actually executes and rebuild the snapshot
    from it — preserving the job's pinned ``variable_values``. If the evaluator
    was deleted, or has no live version, or the live version row is missing, we
    keep the original submit-time snapshot unchanged (a queued job must not break
    because someone edited/deleted the evaluator). The caller writes the refreshed
    list back into the job details so finished-run reads render against the
    live-at-run-time version reproducibly.

    Shared by the STT and TTS runners (`routers/stt.py`, `routers/tts.py`).
    """
    from db import get_evaluator, get_evaluator_version  # local: avoid circular import

    refreshed: List[Dict[str, Any]] = []
    for snap in snapshot_evaluators or []:
        uid = snap.get("uuid")
        evaluator = get_evaluator(uid) if uid else None
        live_id = evaluator.get("live_version_id") if evaluator else None
        version = get_evaluator_version(live_id) if live_id else None
        if not evaluator or not version or version.get("evaluator_id") != uid:
            refreshed.append(snap)
            continue
        refreshed.append(
            {
                "uuid": evaluator["uuid"],
                "name": evaluator["name"],
                "evaluator_type": evaluator.get(
                    "evaluator_type", snap.get("evaluator_type")
                ),
                "data_type": evaluator.get("data_type", snap.get("data_type", "text")),
                "kind": evaluator.get("kind", snap.get("kind", "single")),
                "output_type": evaluator.get(
                    "output_type", snap.get("output_type", "binary")
                ),
                "evaluator_version_id": version["uuid"],
                "judge_model": version["judge_model"],
                "system_prompt": version["system_prompt"],
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
                "variable_values": snap.get("variable_values") or {},
            }
        )
    return refreshed


def build_evaluator_cli_payload_unrendered(
    evaluators: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Same shape as `build_evaluator_cli_payload`, but leaves `{{variable}}`
    placeholders in the system_prompt UNRENDERED so calibrate substitutes per
    test case from each criterion's `arguments`. The rubric block (per-level
    scale descriptions) is still appended.

    Use for the LLM annotation eval flow, where each item supplies its own
    `payload.evaluator_variables[<evaluator_uuid>] = {var: value}` and those
    values become per-test `criteria[].arguments`.
    """
    payload = []
    for ev in evaluators:
        rendered_prompt = ev.get("system_prompt", "")
        rubric = _format_scale_rubric(ev.get("output_config"))
        if rubric:
            rendered_prompt = rendered_prompt.rstrip() + rubric
        payload.append(_calibrate_evaluator_def(ev, rendered_prompt))
    return payload


def build_test_evaluators_payload(
    tests_with_evaluators: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """For LLM tests/benchmarks (deduped top-level evaluators + per-test criteria refs).

    Args:
        tests_with_evaluators: list of {"test_uuid", "evaluators": [evaluator-link-dicts]}
            where each evaluator-link is the row produced by `get_evaluators_for_test`.

    Returns:
        (top_level_evaluators, criteria_per_test)

        - `top_level_evaluators` — unique evaluator definitions (deduped by evaluator uuid).
          Each entry's `system_prompt` is **NOT** rendered with variables; the rubric block
          (per-level scale descriptions) is still appended. Calibrate substitutes
          `{{variable}}` placeholders per test case using each criterion's `arguments`.

        - `criteria_per_test` — `{test_uuid: [{"name", "arguments"?}]}` references that go
          into each test case's `evaluation.criteria`.

    Names must be unique across the run (calibrate keys outputs by name); when two evaluators
    share the same `name`, a `-{shortuuid}` suffix is appended to disambiguate.
    """
    name_for_uuid: Dict[str, str] = {}
    used_names: set = set()
    top_level: List[Dict[str, Any]] = []
    criteria_per_test: Dict[str, List[Dict[str, Any]]] = {}

    for entry in tests_with_evaluators:
        test_uuid = entry["test_uuid"]
        criteria_refs: List[Dict[str, Any]] = []
        for ev in entry.get("evaluators") or []:
            evaluator_uuid = ev.get("uuid")
            if not evaluator_uuid:
                continue

            if evaluator_uuid not in name_for_uuid:
                base_name = ev.get("name") or evaluator_uuid
                name = base_name
                if name in used_names:
                    name = f"{base_name}-{evaluator_uuid[:8]}"
                used_names.add(name)
                name_for_uuid[evaluator_uuid] = name

                # Top-level definition: leave {{variable}} placeholders unrendered so calibrate
                # can substitute per-test arguments. Rubric block IS appended.
                rendered_prompt = ev.get("system_prompt", "")
                rubric = _format_scale_rubric(ev.get("output_config"))
                if rubric:
                    rendered_prompt = rendered_prompt.rstrip() + rubric

                ev_with_name = {**ev, "name": name}
                top_level.append(_calibrate_evaluator_def(ev_with_name, rendered_prompt))

            ref: Dict[str, Any] = {"name": name_for_uuid[evaluator_uuid]}
            arguments = ev.get("variable_values") or {}
            if arguments:
                ref["arguments"] = arguments
            criteria_refs.append(ref)

        criteria_per_test[test_uuid] = criteria_refs

    return top_level, criteria_per_test


