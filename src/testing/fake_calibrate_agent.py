#!/usr/bin/env python3
"""Integration-testing stand-in for the ``calibrate-agent`` eval CLI.

Enabled by ``FAKE_AI_PROVIDERS=1`` (see ``get_calibrate_agent_cli`` in
``src/utils.py``) so integration tests can drive the full run → results pipeline
with no real LLM/STT/TTS call, key, or cost. Standalone — imports nothing from
the backend. For each ``[cli, <subcommand>, ...flags]`` a worker spawns, it
writes the output files that worker's reader expects, then exits 0.

Output shapes mirror the readers in ``routers/agent_tests.py``, ``stt.py``,
``tts.py``, ``simulations.py``, ``annotation_eval_runner.py``, and
``trace_scoring.py``. The constants below are asserted by the frontend
integration tests — keep them stable.

With ``FAKE_AI_DEMO=1`` as well, replies, verdicts, transcripts and numbers come
from ``demo_replies.json`` (an AI maths tutor example used for docs
screenshots) wherever an input matches it; everything else stays canned.
"""

import csv
import json
import os
import statistics
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Canned constants (frontend integration tests assert these) -------------
FAKE_RESPONSE = "Simulated agent reply."
FAKE_REASONING = "Simulated judge reasoning: criteria satisfied."
FAKE_LATENCY_MS = 100
FAKE_COST = 0.001
FAKE_TOKENS = 42
FAKE_WER = 0.0
FAKE_TTFB = 0.5
# Sarvam judge bundle (included by default; suppressed by `--skip-llm-judges`).
FAKE_SARVAM_LLM_WER = 0.0
FAKE_SARVAM_LLM_CER = 0.0
# Every evaluator verdict is a PASS; every rating is scale_max.

# demo_replies.json when FAKE_AI_DEMO=1, else None. Set by `main`.
_DEMO: Optional[Dict[str, Any]] = None


def _demo_test(test_case: Any) -> Optional[Dict[str, Any]]:
    """The ``demo`` block (plus ``expected_tool_call``) of the demo test whose
    ``user_message`` equals the test case's last user turn (or its ``input``
    when it has no history)."""
    if not _DEMO or not isinstance(test_case, dict):
        return None
    text = test_case.get("input")
    for turn in test_case.get("history") or []:
        if isinstance(turn, dict) and turn.get("role") == "user":
            text = turn.get("content")
    if not isinstance(text, str):
        return None
    for t in _DEMO.get("tests") or []:
        if t.get("user_message", "").strip() == text.strip():
            return {**t["demo"], "expected_tool_call": t.get("expected_tool_call")}
    return None


def _write_silent_wav(path: Path, seconds: float = 1.5) -> None:
    """A short valid silent WAV (16 kHz mono) so audio players can load it."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


# --- Argument parsing -------------------------------------------------------
# Flags that take no value (their presence is the signal). Everything else that
# starts with "-" consumes the following non-dash tokens as its value(s).
_BOOL_FLAGS = {"--eval-only", "--skip-verify", "--skip-llm-judges"}


def _parse_args(argv: List[str]) -> tuple[str, Dict[str, List[str]]]:
    """Return (subcommand, {flag: [values]}). Lenient by design — the real CLI
    accepts many more flags; we only need the few the workers pass."""
    sub = argv[1] if len(argv) > 1 else ""
    opts: Dict[str, List[str]] = {}
    i = 2
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-"):
            if tok in _BOOL_FLAGS:
                opts[tok] = []
                i += 1
                continue
            vals: List[str] = []
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                vals.append(argv[i])
                i += 1
            opts[tok] = vals
        else:
            i += 1  # stray positional — ignore
    return sub, opts


def _first(opts: Dict[str, List[str]], *keys: str) -> Optional[str]:
    for k in keys:
        vals = opts.get(k)
        if vals:
            return vals[0]
    return None


def _many(opts: Dict[str, List[str]], *keys: str) -> List[str]:
    for k in keys:
        vals = opts.get(k)
        if vals:
            return vals
    return []


def _load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- Evaluator helpers ------------------------------------------------------
# The calibrate evaluator contract keys the uuid as ``id`` and the output kind
# as ``type`` ("binary" | "rating"), with ``scale_min``/``scale_max`` on rating.
def _evaluators_from_config(config: Any) -> List[Dict[str, Any]]:
    if isinstance(config, dict):
        evs = config.get("evaluators")
        if isinstance(evs, list):
            return [e for e in evs if isinstance(e, dict)]
    return []


def _ev_uuid(ev: Dict[str, Any]) -> Optional[str]:
    return ev.get("id") or ev.get("uuid")


def _ev_scale_max(ev: Dict[str, Any]) -> Any:
    return ev.get("scale_max", 5)


def _ev_scale_min(ev: Dict[str, Any]) -> Any:
    return ev.get("scale_min", 1)


def _ev_value(ev: Dict[str, Any], passed: bool) -> Any:
    """One case's verdict: binary is ``passed``; rating is scale_max on a pass,
    else scale_min + 1."""
    if ev.get("type") == "rating":
        return _ev_scale_max(ev) if passed else _ev_scale_min(ev) + 1
    return passed


def _evaluators_map(evaluators: List[Dict[str, Any]]) -> Dict[str, str]:
    """calibrate's ``config.json`` map is ``{evaluator_uuid: name}``."""
    out: Dict[str, str] = {}
    for ev in evaluators:
        uid = _ev_uuid(ev)
        name = ev.get("name")
        if uid and name:
            out[str(uid)] = str(name)
    return out


def _write_config_json(output_dir: Path, evaluators: List[Dict[str, Any]]) -> None:
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"evaluators_map": _evaluators_map(evaluators)}, f)


def _metrics_aggregate(
    ev: Dict[str, Any], n: int, pass_rate: float = 100.0
) -> Dict[str, Any]:
    """Per-evaluator aggregate entry for metrics.json. Must carry a ``type`` key
    so the readers recognise it as an evaluator output (not a built-in scalar).
    Below 100, a failed case scores ``scale_min + 1`` on a rating evaluator,
    matching the per-case demo scores."""
    uid = _ev_uuid(ev)
    if ev.get("type") == "rating":
        sm = _ev_scale_max(ev)
        low = _ev_value(ev, pass_rate >= 100)
        return {
            "type": "rating",
            "evaluator_id": uid,
            "mean": sm if pass_rate >= 100 else low + (sm - low) * pass_rate / 100,
            "min": low,
            "max": sm,
            "count": n,
            "scale_min": _ev_scale_min(ev),
            "scale_max": sm,
        }
    return {
        "type": "binary",
        # Out of 100, as the real CLI writes it: (passed / total) * 100.
        "pass_rate": pass_rate,
        "evaluator_id": uid,
        "passed": round(n * pass_rate / 100),
        "total": n,
    }


# --- Subcommand: llm (unit-test run AND benchmark) --------------------------
def _safe_model(model: str) -> str:
    """Filesystem-safe folder name matching ``_match_model_to_folder``."""
    return model.replace("/", "__").replace(":", "_")


def _llm_judge_results(
    criteria: List[Dict[str, Any]],
    ev_by_name: Dict[str, Dict[str, Any]],
    demo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    passed = demo["passed"] if demo else True
    reasoning = demo["reasoning"] if demo else FAKE_REASONING
    judge_results: Dict[str, Any] = {}
    for ref in criteria:
        if not isinstance(ref, dict):
            continue
        name = ref.get("name")
        if not name:
            continue
        ev = ev_by_name.get(name, {})
        judgement = {"reasoning": reasoning, "evaluator_id": _ev_uuid(ev)}
        judgement["score" if ev.get("type") == "rating" else "match"] = _ev_value(ev, passed)
        judge_results[name] = judgement
    return judge_results


def _write_llm_model_dir(
    model_dir: Path,
    test_cases: List[Dict[str, Any]],
    evaluators: List[Dict[str, Any]],
    model: str = "default",
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    ev_by_name = {ev.get("name"): ev for ev in evaluators if ev.get("name")}
    bm = next(
        (b for b in (_DEMO or {}).get("benchmark_models") or [] if b.get("model") == model),
        None,
    )

    results: List[Dict[str, Any]] = []
    ref_count: Dict[str, int] = {}
    for tc in test_cases:
        if not isinstance(tc, dict):
            continue
        evaluation = tc.get("evaluation") or {}
        ev_type = evaluation.get("type")
        criteria = evaluation.get("criteria") or []
        demo = _demo_test(tc)
        # A model at 100% passes every case, so its failing cases use the pass reply.
        if demo and not demo["passed"] and bm and bm["pass_rate"] >= 100 and demo.get("pass_reply"):
            demo = {"reply": demo["pass_reply"], "passed": True, "reasoning": demo["pass_reasoning"]}
        judge_results = _llm_judge_results(criteria, ev_by_name, demo)
        for name in judge_results:
            ref_count[name] = ref_count.get(name, 0) + 1

        output: Dict[str, Any] = {
            "response": demo["reply"] if demo else FAKE_RESPONSE,
            "tool_calls": [],
            "cost": FAKE_COST,
        }
        if demo and demo.get("expected_tool_call"):
            # The real CLI's output shape: {"tool", "arguments"} with plain values.
            call = demo["expected_tool_call"]
            output["tool_calls"] = [{"tool": call["name"], "arguments": call["arguments"]}]
        elif ev_type == "tool_call":
            output["tool_calls"] = evaluation.get("tool_calls") or []

        results.append(
            {
                "test_case_id": tc.get("id"),
                "test_case": tc,
                "output": output,
                "metrics": {
                    "passed": demo["passed"] if demo else True,
                    "reasoning": demo["reasoning"] if demo else FAKE_REASONING,
                    "judge_results": judge_results,
                },
                "latency_ms": FAKE_LATENCY_MS,
            }
        )

    n = len(results)
    passed = sum(1 for r in results if r["metrics"]["passed"])
    pass_rate = 100.0 * passed / n if n else 100.0
    latency, cost = FAKE_LATENCY_MS, FAKE_COST
    if bm:
        # A model listed in benchmark_models reports its demo latency and cost.
        latency, cost = bm["latency_ms"], bm["cost"]
        for r in results:
            r["latency_ms"] = latency
            r["output"]["cost"] = cost

    with open(model_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

    criteria_agg: Dict[str, Any] = {}
    for ev in evaluators:
        name = ev.get("name")
        if not name:
            continue
        criteria_agg[name] = _metrics_aggregate(ev, ref_count.get(name, n), pass_rate)

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": n,
                "passed": passed,
                # Aggregate perf blocks are dicts, not scalars: latency is
                # percentiles {p50,p95,p99,count}; cost/tokens are
                # {mean,min,max,count}. The run-status/benchmark readers pass
                # these straight into Optional[Dict] response fields, so a
                # scalar 500s response validation.
                "latency_ms": {"p50": latency, "p95": latency, "p99": latency, "count": n},
                "cost": {"mean": cost, "min": cost, "max": cost, "count": n},
                "total_tokens": {
                    "mean": FAKE_TOKENS,
                    "min": FAKE_TOKENS,
                    "max": FAKE_TOKENS,
                    "count": n,
                },
                # Every case answered, so no gaps and no early stop.
                "errored": 0,
                "stopped_early": False,
                "criteria": criteria_agg,
            },
            f,
        )


def _write_leaderboard(
    output_dir: Path, safe_models: List[str], evaluators: List[Dict[str, Any]]
) -> None:
    leaderboard_dir = output_dir / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    ev_names = [ev.get("name") for ev in evaluators if ev.get("name")]
    header = ["model", "test_pass_rate"] + ev_names
    with open(leaderboard_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sm in safe_models:
            writer.writerow([sm, "1.0"] + ["1.0" for _ in ev_names])


def _write_demo_leaderboard(output_dir: Path, safe_models: List[str]) -> None:
    """Leaderboard in the real CLI's columns (``tests_leaderboard.py``), built
    from each model's metrics.json, so the benchmark page shows the demo
    pass rate, latency and cost per model."""
    leaderboard_dir = output_dir / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        sm: json.loads((output_dir / sm / "metrics.json").read_text(encoding="utf-8"))
        for sm in safe_models
    }
    names = sorted({c for m in metrics.values() for c in m["criteria"]})
    with open(leaderboard_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "passed", "total", "pass_rate", "latency_p50", "latency_p95",
             "latency_p99", "cost", "total_tokens"] + names
        )
        for sm, m in metrics.items():
            crit = [m["criteria"].get(c, {}) for c in names]
            writer.writerow(
                [sm, m["passed"], m["total"],
                 100 * m["passed"] / m["total"] if m["total"] else "",
                 m["latency_ms"]["p50"], m["latency_ms"]["p95"], m["latency_ms"]["p99"],
                 m["cost"]["mean"], m["total_tokens"]["mean"]]
                + [c.get("mean") if c.get("type") == "rating" else c.get("pass_rate") for c in crit]
            )


def _cmd_llm(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    config = _load_json(_first(opts, "-c", "--config"))
    test_cases = []
    if isinstance(config, dict):
        test_cases = [t for t in (config.get("test_cases") or []) if isinstance(t, dict)]
    evaluators = _evaluators_from_config(config)

    models = _many(opts, "-m", "--model", "--models")
    if not models:
        # Agent-connection unit-test mode: no -m. The unit-test reader walks the
        # tree for the first results.json, so a single folder name is enough.
        models = ["default"]

    output_dir.mkdir(parents=True, exist_ok=True)
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)
    safe_models: List[str] = []
    for model in models:
        safe = _safe_model(model) if model != "default" else "default"
        safe_models.append(safe)
        _write_llm_model_dir(output_dir / safe, test_cases, evaluators, model)

    # Benchmark reads a leaderboard; a unit test ignores it. Harmless either way.
    if _many(opts, "-m", "--model", "--models"):
        if _DEMO:
            _write_demo_leaderboard(output_dir, safe_models)
        else:
            _write_leaderboard(output_dir, safe_models, evaluators)



# --- Subcommand: stt --------------------------------------------------------
def _read_id_text_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except OSError:
        pass
    return rows


def _evaluator_row_cols(
    evaluators: List[Dict[str, Any]],
) -> tuple[List[str], Dict[str, str]]:
    """Column names + a per-column canned value for evaluator outputs in a
    results.csv row (``<name>`` score + ``<name>_reasoning``)."""
    cols: List[str] = []
    values: Dict[str, str] = {}
    for ev in evaluators:
        name = ev.get("name")
        if not name:
            continue
        cols.append(name)
        cols.append(f"{name}_reasoning")
        values[name] = str(_ev_scale_max(ev)) if ev.get("type") == "rating" else "1"
        values[f"{name}_reasoning"] = FAKE_REASONING
    return cols, values


# Extra scalar metrics + per-row columns the real CLI writes for the Sarvam
# judge bundle (on by default). Mirrors `_score_and_write_results` in calibrate-agent.
_SARVAM_METRICS = {"sarvam_llm_wer": FAKE_SARVAM_LLM_WER, "sarvam_llm_cer": FAKE_SARVAM_LLM_CER}
_SARVAM_ROW_COLS = ["sarvam_llm_wer", "sarvam_llm_cer", "sarvam_llm_wer_reasoning"]
_SARVAM_ROW_VALUES = {
    "sarvam_llm_wer": str(FAKE_SARVAM_LLM_WER),
    "sarvam_llm_cer": str(FAKE_SARVAM_LLM_CER),
    "sarvam_llm_wer_reasoning": "[]",
}


def _stt_metrics(
    evaluators: List[Dict[str, Any]], sarvam: bool, wer: float = FAKE_WER
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"wer": wer}
    if sarvam:
        metrics.update(_SARVAM_METRICS)
    for ev in evaluators:
        name = ev.get("name")
        if name:
            metrics[name] = _metrics_aggregate(ev, 1)
    return metrics


def _cmd_stt(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = _many(opts, "-p", "--provider", "--providers") or ["openai"]
    evaluators = _evaluators_from_config(_load_json(_first(opts, "--config")))
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)
    sarvam = "--skip-llm-judges" not in opts

    input_dir = _first(opts, "-i", "--input")
    utterances = _read_id_text_csv(Path(input_dir) / "stt.csv") if input_dir else []
    if not utterances:
        utterances = [{"id": "audio_1", "text": "x"}]

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    extra_cols = _SARVAM_ROW_COLS if sarvam else []
    extra_values = _SARVAM_ROW_VALUES if sarvam else {}
    for provider in providers:
        sub = output_dir / f"{provider}_results"
        sub.mkdir(parents=True, exist_ok=True)
        wer = ((_DEMO or {}).get("stt_dataset") or {}).get("demo_wer", {}).get(provider, FAKE_WER)
        k, ok, bad, judge = len(utterances), ev_values, ev_values, {}
        nums: Dict[str, float] = {}
        if _DEMO:
            k, ok, bad, judge = _demo_judge("stt_dataset", provider, evaluators, len(utterances))
            d = _DEMO.get("stt_dataset") or {}
            nums = {"wer": wer}
            for key in ("cer", "sarvam_llm_wer", "sarvam_llm_cer"):
                value = (d.get(f"demo_{key}") or {}).get(provider)
                if value is not None and (sarvam or key == "cer"):
                    nums[key] = value
        num_cols = [c for c in ("wer", "cer") if c in nums]
        row_extra = {c: str(nums.get(c, v)) for c, v in extra_values.items()}
        with open(sub / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "gt", "pred"] + num_cols + ev_cols + extra_cols)
            for idx, utt in enumerate(utterances):
                text = utt.get("text", "")
                cells = ok if idx < k else bad
                writer.writerow(
                    [utt.get("id"), text, text]
                    + [nums[c] for c in num_cols]
                    + [cells[c] for c in ev_cols]
                    + [row_extra[c] for c in extra_cols]
                )
        metrics = _stt_metrics(evaluators, sarvam, wer)
        metrics.update(nums)
        metrics.update(judge)
        with open(sub / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)
    if _DEMO:
        _write_demo_speech_leaderboard(output_dir, providers, len(utterances), "stt_leaderboard.xlsx")



def _demo_judge(
    dataset: str, provider: str, evaluators: List[Dict[str, Any]], n: int
) -> tuple[int, Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Demo judge outcome for one STT/TTS provider: how many rows pass, the
    evaluator cells for a passing and a failing row, and the metrics.json
    entries. Entries carry the real CLI's ``mean`` (binary as a 0-1 fraction),
    which the leaderboard's evaluator columns read."""
    d = _DEMO.get(dataset) or {}
    k = round((d.get("demo_judge_pass_rate") or {}).get(provider, 1.0) * n)
    ok: Dict[str, str] = {}
    bad: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    for ev in evaluators:
        name = ev.get("name")
        if not name:
            continue
        ok[name] = str(int(_ev_value(ev, True)))
        bad[name] = str(int(_ev_value(ev, False)))
        ok[f"{name}_reasoning"] = d.get("demo_judge_reasoning_pass", FAKE_REASONING)
        bad[f"{name}_reasoning"] = d.get("demo_judge_reasoning_fail", FAKE_REASONING)
        agg = _metrics_aggregate(ev, n, 100.0 * k / n if n else 100.0)
        agg.setdefault("mean", k / n if n else 1.0)
        metrics[name] = agg
    return k, ok, bad, metrics


def _write_demo_speech_leaderboard(
    output_dir: Path, providers: List[str], count: int, filename: str
) -> None:
    """The ``summary`` sheet the real CLI's ``{stt,tts}/leaderboard.py`` writes
    (one row per provider: ``run``, ``count``, then the metrics.json numbers
    flattened as ``read_leaderboard_metrics`` does), read by
    ``read_leaderboard_xlsx``. The real CLI names provider folders by the bare
    provider, so ``run`` is the provider."""
    import openpyxl  # in the backend env; only demo mode needs it

    rows = []
    for provider in providers:
        metrics = json.loads(
            (output_dir / f"{provider}_results" / "metrics.json").read_text(encoding="utf-8")
        )
        row: Dict[str, Any] = {"run": provider, "count": count}
        for key, value in metrics.items():
            if isinstance(value, dict) and "mean" in value:
                row[key] = value["mean"]
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = float(value)
        rows.append(row)
    headers = list(dict.fromkeys(k for r in rows for k in r))
    (output_dir / "leaderboard").mkdir(exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    wb.save(output_dir / "leaderboard" / filename)


# --- Subcommand: tts --------------------------------------------------------
def _tts_metrics(evaluators: List[Dict[str, Any]], ttfb: float = FAKE_TTFB) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"ttfb": ttfb}
    for ev in evaluators:
        name = ev.get("name")
        if name:
            metrics[name] = _metrics_aggregate(ev, 1)
    return metrics


def _cmd_tts(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = _many(opts, "-p", "--provider", "--providers") or ["openai"]
    evaluators = _evaluators_from_config(_load_json(_first(opts, "--config")))
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)

    input_file = _first(opts, "-i", "--input")
    rows = _read_id_text_csv(Path(input_file)) if input_file else []
    if not rows:
        rows = [{"id": "row_1", "text": "hi"}]

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    for provider in providers:
        k, ok, bad, judge = len(rows), ev_values, ev_values, {}
        if _DEMO:
            k, ok, bad, judge = _demo_judge("tts_dataset", provider, evaluators, len(rows))
        sub = output_dir / f"{provider}_results"
        audios = sub / "audios"
        audios.mkdir(parents=True, exist_ok=True)
        with open(sub / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "audio_path"] + ev_cols)
            for idx, row in enumerate(rows):
                # Write a real .wav and reference its ABSOLUTE path — the
                # TTS worker keys audio files by exact local path and marks a
                # provider failed if no row's audio_path matches a walked file.
                wav = audios / f"row_{idx + 1}.wav"
                _write_silent_wav(wav)
                writer.writerow(
                    [row.get("id"), row.get("text", ""), str(wav)]
                    + [(ok if idx < k else bad)[c] for c in ev_cols]
                )
        ttfb = ((_DEMO or {}).get("tts_dataset") or {}).get("demo_ttfb_seconds", {}).get(provider, FAKE_TTFB)
        metrics = _tts_metrics(evaluators, ttfb)
        metrics.update(judge)
        with open(sub / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)
    if _DEMO:
        _write_demo_speech_leaderboard(output_dir, providers, len(rows), "tts_leaderboard.xlsx")



# --- Subcommand: simulations (normal run) -----------------------------------
def _sim_eval_row(ev: Dict[str, Any]) -> List[Any]:
    if ev.get("type") == "rating":
        return [_ev_uuid(ev), ev.get("name"), "rating", _ev_scale_max(ev), FAKE_REASONING]
    return [_ev_uuid(ev), ev.get("name"), "binary", "Pass", FAKE_REASONING]


def _demo_eval_row(
    ev: Dict[str, Any], passed: bool, reasoning: str, score: Any = None
) -> List[Any]:
    """An evaluation_results.csv row in the real CLI's value shape: binary
    1.0/0.0, rating the score (``score`` when given) as a float."""
    kind = "rating" if ev.get("type") == "rating" else "binary"
    value = float(_ev_value(ev, passed) if score is None else score)
    return [_ev_uuid(ev), ev.get("name"), kind, value, reasoning]


def _write_sim_case(
    case_dir: Path,
    persona: Any,
    scenario: Any,
    evaluators: List[Dict[str, Any]],
    voice: bool = False,
) -> List[List[Any]]:
    """Writes one case and returns its evaluation rows."""
    case_dir.mkdir(parents=True, exist_ok=True)
    transcript = [
        {"role": "user", "content": "Simulated user turn."},
        {"role": "assistant", "content": FAKE_RESPONSE},
    ]
    rows = [_sim_eval_row(ev) for ev in evaluators]
    if _DEMO and isinstance(persona, dict) and isinstance(scenario, dict):
        sim = _DEMO["simulation"]
        key = f"{persona.get('label')}|{scenario.get('name')}"
        transcript = sim["demo_transcripts"].get(key, transcript)
        verdicts = sim.get("demo_verdicts", {}).get(key, {})
        rows = []
        for ev in evaluators:
            # An evaluator listed in demo_verdicts fails with its own reasoning.
            v = verdicts.get(ev.get("name"))
            if v:
                rows.append(_demo_eval_row(ev, v.get("pass", False), v["reasoning"], v.get("score")))
            else:
                rows.append(_demo_eval_row(ev, True, sim["demo_reasoning_pass"]))
        if voice:
            # Per-case timings the real voice CLI appends; the voice results
            # page averages them per run.
            rows += [["", k, "", v, ""] for k, v in sim["demo_voice_latency_seconds"].items()]
            # Turn audio is named <exchange>_<bot|user>.wav, exchange N
            # holding spoken turns 2N-2 and 2N-1 (the order the reader sorts by).
            audios = case_dir / "audios"
            audios.mkdir(exist_ok=True)
            for i, turn in enumerate(transcript):
                role = "bot" if turn["role"] == "assistant" else "user"
                _write_silent_wav(audios / f"{i // 2 + 1}_{role}.wav")
            _write_silent_wav(case_dir / "conversation.wav", 1.5 * len(transcript))
    with open(case_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f)
    with open(case_dir / "evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluator_id", "name", "type", "value", "reasoning"])
        writer.writerows(rows)
    with open(case_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"persona": persona, "scenario": scenario}, f)
    return rows


def _demo_sim_metrics(
    evaluators: List[Dict[str, Any]], rows: List[List[Any]]
) -> Dict[str, Any]:
    """Real-CLI metrics.json shape (``summarize_metric_distribution``)."""
    metrics: Dict[str, Any] = {}
    for ev in evaluators:
        name = ev.get("name")
        values = [r[3] for r in rows if r[1] == name]
        if not name or not values:
            continue
        entry: Dict[str, Any] = {
            "type": ev.get("type") or "binary",
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
        if ev.get("type") == "rating":
            entry["scale_min"], entry["scale_max"] = _ev_scale_min(ev), _ev_scale_max(ev)
        entry["evaluator_id"] = _ev_uuid(ev)
        metrics[name] = entry
    return metrics


def _cmd_simulations(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_json(_first(opts, "-c", "--config")) or {}
    personas = config.get("personas") or []
    scenarios = config.get("scenarios") or []
    evaluators = _evaluators_from_config(config)
    voice = _first(opts, "--type") == "voice"

    all_rows: List[List[Any]] = []
    for i, persona in enumerate(personas, start=1):
        for j, scenario in enumerate(scenarios, start=1):
            case_dir = output_dir / f"simulation_persona_{i}_scenario_{j}"
            all_rows += _write_sim_case(case_dir, persona, scenario, evaluators, voice)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        if _DEMO:
            json.dump(_demo_sim_metrics(evaluators, all_rows), f)
        else:
            json.dump({"total": len(personas) * len(scenarios)}, f)


# --- Annotation eval-only flows ---------------------------------------------
def _cmd_annotation_tts(opts: Dict[str, List[str]]) -> None:
    """`tts --eval-only` reads a prior run dir's ``results.csv`` (id, text,
    audio_path) and writes judged ``results.csv`` + ``metrics.json`` under
    ``-o`` — same flat layout as STT eval-only, but with TTS base columns."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(_first(opts, "--dataset"))
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)

    csv_path = run_dir / "results.csv"
    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "audio_path"] + ev_cols)
        for row in rows:
            writer.writerow(
                [row.get("id"), row.get("text"), row.get("audio_path")]
                + [ev_values[c] for c in ev_cols]
            )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_tts_metrics(evaluators), f)


def _cmd_annotation_stt_or_general(opts: Dict[str, List[str]]) -> None:
    """`stt --eval-only` and `general` both write a flat run-root results.csv
    (id + built-in cols + per-evaluator score/reasoning) keyed by dataset id."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)
    dataset = _load_json(_first(opts, "--dataset")) or []

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred"] + ev_cols)
        for row in dataset if isinstance(dataset, list) else []:
            if not isinstance(row, dict):
                continue
            gt = row.get("gt", row.get("input", ""))
            pred = row.get("pred", row.get("output", ""))
            writer.writerow(
                [row.get("id"), gt, pred] + [ev_values[c] for c in ev_cols]
            )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        # Sarvam judges are an STT-eval-only feature; the annotation eval-only
        # and `general` flows never include the Sarvam bundle.
        json.dump(_stt_metrics(evaluators, sarvam=False), f)


def _cmd_annotation_llm(opts: Dict[str, List[str]]) -> None:
    """`llm --eval-only` writes a flat run-root results.json keyed by
    ``test_case_id``, with judge_results per referenced evaluator."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)
    ev_by_name = {ev.get("name"): ev for ev in evaluators if ev.get("name")}
    dataset = _load_json(_first(opts, "--dataset")) or []

    results: List[Dict[str, Any]] = []
    for row in dataset if isinstance(dataset, list) else []:
        if not isinstance(row, dict):
            continue
        test_case = row.get("test_case") or {}
        criteria = (test_case.get("evaluation") or {}).get("criteria") or []
        demo = _demo_test(test_case)
        judge_results = _llm_judge_results(criteria, ev_by_name, demo)
        results.append(
            {
                "test_case_id": test_case.get("id"),
                "test_case": test_case,
                "output": {
                    "response": demo["reply"] if demo else FAKE_RESPONSE,
                    "tool_calls": [],
                    "cost": FAKE_COST,
                },
                "metrics": {
                    "passed": demo["passed"] if demo else True,
                    "reasoning": demo["reasoning"] if demo else FAKE_REASONING,
                    "judge_results": judge_results,
                },
                "latency_ms": FAKE_LATENCY_MS,
            }
        )

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        passed = sum(1 for r in results if r["metrics"]["passed"])
        json.dump({"total": len(results), "passed": passed}, f)


def _cmd_annotation_simulation(opts: Dict[str, List[str]]) -> None:
    """`simulations --eval-only` writes one ``row_<i>/`` dir per dataset row plus
    a ``dataset_map.json`` mapping ``row_<i>`` → {index, name}."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    # Written before any results file: the real CLI does the same, and
    # `annotation_eval_runner` waits for this file before it reads a
    # results file that is still being appended to.
    _write_config_json(output_dir, evaluators)
    dataset = _load_json(_first(opts, "--dataset")) or []

    dataset_map: Dict[str, Any] = {}
    for idx, row in enumerate(dataset if isinstance(dataset, list) else []):
        row_id = f"row_{idx + 1}"
        name = row.get("name") if isinstance(row, dict) else None
        dataset_map[row_id] = {"index": idx, "name": name}
        case_dir = output_dir / row_id
        case_dir.mkdir(parents=True, exist_ok=True)
        # A demo item is one test exchange; its student message picks the verdict.
        demo = _demo_test({"history": row.get("conversation_history")}) if isinstance(row, dict) else None
        with open(case_dir / "evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["evaluator_id", "name", "type", "value", "reasoning"])
            for ev in evaluators:
                writer.writerow(
                    _demo_eval_row(ev, demo["passed"], demo["reasoning"]) if demo else _sim_eval_row(ev)
                )

    with open(output_dir / "dataset_map.json", "w", encoding="utf-8") as f:
        json.dump(dataset_map, f)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"total": len(dataset_map)}, f)


# --- status (provider health) -----------------------------------------------
def _cmd_status() -> None:
    """Placeholder healthy payload. `provider_status.run_check` short-circuits
    `status` under the flag (returning `_fake_healthy_providers()`), so the fake
    is never spawned for it; this stays only to keep `status` a handled subcommand."""
    sys.stdout.write(json.dumps({"openai": {"status": "pass"}}))


# Every `calibrate-agent <subcommand>` the backend workers spawn. The guard test
# `tests/test_fake_matches_real_usage.py` statically scans the real call sites and
# fails if any spawns a subcommand missing from this set — so a new real usage
# can't ship without the fake growing a handler for it.
SUPPORTED_SUBCOMMANDS = frozenset(
    {"status", "general", "llm", "stt", "tts", "simulations"}
)


def main(argv: List[str]) -> int:
    global _DEMO
    _DEMO = None
    if os.environ.get("FAKE_AI_DEMO") == "1":
        with open(Path(__file__).with_name("demo_replies.json"), encoding="utf-8") as f:
            _DEMO = json.load(f)
    sub, opts = _parse_args(argv)
    eval_only = "--eval-only" in opts

    if sub not in SUPPORTED_SUBCOMMANDS:
        sys.stderr.write(f"fake_calibrate_agent: unknown subcommand {sub!r}\n")
        return 1

    if sub == "status":
        _cmd_status()
    elif sub == "general":
        _cmd_annotation_stt_or_general(opts)
    elif sub == "llm":
        _cmd_annotation_llm(opts) if eval_only else _cmd_llm(opts)
    elif sub == "stt":
        _cmd_annotation_stt_or_general(opts) if eval_only else _cmd_stt(opts)
    elif sub == "tts":
        _cmd_annotation_tts(opts) if eval_only else _cmd_tts(opts)
    elif sub == "simulations":
        _cmd_annotation_simulation(opts) if eval_only else _cmd_simulations(opts)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
