"""Usage numbers across every workspace, for the superadmin Analytics page.

A handful of grouped queries (by workspace and day) feed the totals, the daily
chart, the per-workspace table, and the lists of failures and limit hits.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from db import get_db_connection

# Run kind -> the field that counts it.
EVAL_FIELDS = {
    "test_run": "test_runs",
    "model_comparison": "model_comparisons",
    "simulation": "simulations",
    "stt": "stt",
    "tts": "tts",
    "labelling_evaluator_run": "labelling_evaluator_runs",
}
EVAL_KINDS = tuple(EVAL_FIELDS.values())
# What makes a workspace count as active on a day.
ACTIVE_METRICS = set(EVAL_KINDS) | {"traces_received", "labels_collected"}
DAY_METRICS = EVAL_KINDS + (
    "failed_evaluations",
    "traces_received",
    "traces_scored",
    "labels_collected",
    "labelling_jobs_finished",
)
RANGE_METRICS = DAY_METRICS + ("limit_hits",)
LIST_LIMIT = 100
ALL_TIME_START = "0000-01-01"

# Not a real user yet: an invite stub, or a signup whose email link was never opened.
_STUB_USER = "(u.email_verified_at IS NULL)"

# Workspaces that count: not deleted, and not the personal workspace of a stub user.
_VALID_ORGS = f"""
valid_orgs AS (
    SELECT o.uuid, o.name, o.is_personal
      FROM organizations o
      LEFT JOIN users u ON u.uuid = o.created_by_user_id
     WHERE o.deleted_at IS NULL
       AND NOT (o.is_personal = 1 AND COALESCE({_STUB_USER}, 0))
)"""

# Error text, read only for failed rows. Older runs stored true/false, which reads as null.
_ERR = """CASE WHEN j.status = 'failed' AND json_valid(j.results)
               AND json_type(j.results, '$.error') = 'text'
              THEN json_extract(j.results, '$.error') END"""

# Every evaluation run, one row each.
_EVALS = """
evals AS (
    SELECT a.org_uuid, j.uuid AS id, j.created_at AS ts, j.status, j.cost,
           CASE j.type WHEN 'llm-unit-test' THEN 'test_run' ELSE 'model_comparison' END AS kind,
           {err} AS error
      FROM agent_test_jobs j JOIN agents a ON a.uuid = j.agent_id
     WHERE j.type IN ('llm-unit-test', 'llm-benchmark')
    UNION ALL
    SELECT s.org_uuid, j.uuid, j.created_at, j.status, NULL, 'simulation',
           {err}
      FROM simulation_jobs j JOIN simulations s ON s.uuid = j.simulation_id
    UNION ALL
    SELECT j.org_uuid, j.uuid, j.created_at, j.status, NULL,
           CASE j.type WHEN 'stt-eval' THEN 'stt' WHEN 'tts-eval' THEN 'tts'
                ELSE 'labelling_evaluator_run' END,
           {err}
      FROM jobs j
     WHERE j.deleted_at IS NULL AND j.type IN ('stt-eval', 'tts-eval', 'annotation-eval')
)""".format(err=_ERR)

# Every trace sent after the scoring cap is left unscored with its own row.
# Count those as one hit per workspace per hour, the same rule the trace cap
# uses when it records its own hits.
_SCORING_CAP_HITS = """
    SELECT org_uuid, MIN(created_at) AS created_at FROM trace_eval_runs
     WHERE status = 'skipped' AND error = 'over_limit'
     GROUP BY org_uuid, strftime('%Y-%m-%d %H', datetime(created_at))"""

# Everything counted per workspace per day: (org_uuid, ts, metric, cost).
_EVENTS = """
events AS (
    SELECT org_uuid, ts, kind AS metric, cost FROM evals
    UNION ALL SELECT org_uuid, ts, 'failed_evaluations', NULL FROM evals WHERE status = 'failed'
    UNION ALL SELECT org_uuid, created_at, 'traces_received', NULL
                FROM traces WHERE deleted_at IS NULL
    UNION ALL SELECT r.org_uuid, r.completed_at, 'traces_scored', NULL
                FROM trace_eval_runs r JOIN traces tr ON tr.uuid = r.trace_uuid
               WHERE r.status = 'completed' AND tr.deleted_at IS NULL
    UNION ALL SELECT t.org_uuid, j.completed_at, 'labelling_jobs_finished', NULL
                FROM annotation_jobs j JOIN annotation_tasks t ON t.uuid = j.task_id
               WHERE j.status = 'completed' AND j.deleted_at IS NULL AND t.deleted_at IS NULL
    UNION ALL SELECT t.org_uuid, an.created_at, 'labels_collected', NULL
                FROM annotations an
                JOIN annotation_jobs j ON j.uuid = an.job_id
                JOIN annotation_tasks t ON t.uuid = j.task_id
               WHERE an.deleted_at IS NULL AND j.deleted_at IS NULL AND t.deleted_at IS NULL
    UNION ALL SELECT org_uuid, created_at, 'limit_hits', NULL FROM limit_hits
    UNION ALL SELECT org_uuid, created_at, 'limit_hits', NULL FROM ({cap_hits})
)""".format(cap_hits=_SCORING_CAP_HITS)

_DAY = "date(datetime({col}, :tz))"

_ISO = "strftime('%Y-%m-%dT%H:%M:%SZ', datetime({col}))"


def _empty_activity() -> dict:
    out = {m: 0 for m in RANGE_METRICS}
    out.update(
        new_users=0, active_workspaces=0, evaluations=0, cost=None, items_labelled=0
    )
    return out


def _add_row(act: dict, r) -> None:
    """Add one grouped row's count, and its cost when it has one."""
    act[r["metric"]] += r["n"]
    if r["cost"] is not None:
        act["cost"] = (act["cost"] or 0) + r["cost"]


def _summarise(rows, new_users_by_day, items_by_org, lo: str, hi: str) -> dict:
    """Add up the grouped rows whose day falls in [lo, hi]."""
    act = _empty_activity()
    active = set()
    for r in rows:
        if not lo <= r["day"] <= hi:
            continue
        _add_row(act, r)
        if r["metric"] in ACTIVE_METRICS:
            active.add(r["org_uuid"])
    act["evaluations"] = sum(act[k] for k in EVAL_KINDS)
    act["active_workspaces"] = len(active)
    act["new_users"] = sum(n for d, n in new_users_by_day.items() if lo <= d <= hi)
    act["items_labelled"] = sum(items_by_org.values())
    return act


def get_analytics(start: Optional[date], end: date, tz_offset_minutes: int) -> dict:
    hi = end.isoformat()
    lo = start.isoformat() if start else ALL_TIME_START
    prev_lo = prev_hi = None
    if start:
        length = (end - start).days + 1
        prev_lo = (start - timedelta(days=length)).isoformat()
        prev_hi = (start - timedelta(days=1)).isoformat()
    params = {"tz": f"{tz_offset_minutes} minutes", "lo": lo, "hi": hi, "plo": prev_lo or lo}

    with get_db_connection() as conn:
        q = lambda sql, extra=None: conn.execute(sql, {**params, **(extra or {})}).fetchall()

        users_total = q(f"SELECT COUNT(*) FROM users u WHERE NOT {_STUB_USER}")[0][0]
        orgs = q(
            f"""WITH {_VALID_ORGS}
                SELECT v.uuid, v.name, v.is_personal,
                       (SELECT COUNT(*) FROM organization_members m JOIN users u ON u.uuid = m.user_id
                         WHERE m.org_uuid = v.uuid AND m.deleted_at IS NULL AND NOT {_STUB_USER}) AS members
                  FROM valid_orgs v"""
        )
        running = q(
            f"""WITH {_VALID_ORGS}, {_EVALS}
                SELECT COUNT(*) FROM evals
                 WHERE status IN ('queued', 'in_progress')
                   AND org_uuid IN (SELECT uuid FROM valid_orgs)"""
        )[0][0]

        # One row per (workspace, day, metric) from the start of the previous
        # period to the end of this one.
        rows = q(
            f"""WITH {_VALID_ORGS}, {_EVALS}, {_EVENTS}
                SELECT org_uuid, day, metric, COUNT(*) AS n, SUM(cost) AS cost
                  FROM (SELECT org_uuid, metric, cost, {_DAY.format(col="ts")} AS day FROM events)
                 WHERE day BETWEEN :plo AND :hi AND org_uuid IN (SELECT uuid FROM valid_orgs)
                 GROUP BY org_uuid, day, metric"""
        )
        new_users_by_day = {
            r["day"]: r["n"]
            for r in q(
                f"""SELECT day, COUNT(*) AS n
                      FROM (SELECT {_DAY.format(col="u.created_at")} AS day FROM users u
                             WHERE NOT {_STUB_USER})
                     WHERE day BETWEEN :plo AND :hi GROUP BY day"""
            )
        }

        def items_labelled(a: str, b: str) -> dict:
            return {
                r["org_uuid"]: r["n"]
                for r in q(
                    f"""WITH {_VALID_ORGS}
                        SELECT t.org_uuid, COUNT(DISTINCT an.item_id) AS n
                          FROM annotations an
                          JOIN annotation_jobs j ON j.uuid = an.job_id
                          JOIN annotation_tasks t ON t.uuid = j.task_id
                         WHERE an.deleted_at IS NULL AND j.deleted_at IS NULL AND t.deleted_at IS NULL
                           AND {_DAY.format(col="an.created_at")} BETWEEN :a AND :b
                           AND t.org_uuid IN (SELECT uuid FROM valid_orgs)
                         GROUP BY t.org_uuid""",
                    {"a": a, "b": b},
                )
            }

        items_now = items_labelled(lo, hi)
        items_prev = items_labelled(prev_lo, prev_hi) if start else {}

        last_active = {
            r["org_uuid"]: r["at"]
            for r in q(
                f"""WITH {_EVALS}
                    SELECT org_uuid, {_ISO.format(col="MAX(datetime(ts))")} AS at FROM (
                        SELECT org_uuid, ts FROM evals
                        UNION ALL SELECT org_uuid, created_at FROM traces WHERE deleted_at IS NULL
                        UNION ALL SELECT t.org_uuid, an.created_at FROM annotations an
                                    JOIN annotation_jobs j ON j.uuid = an.job_id
                                    JOIN annotation_tasks t ON t.uuid = j.task_id
                                   WHERE an.deleted_at IS NULL AND j.deleted_at IS NULL AND t.deleted_at IS NULL)
                     GROUP BY org_uuid"""
            )
        }

        in_range = f"""org_uuid IN (SELECT uuid FROM valid_orgs)
                       AND {_DAY.format(col="ts")} BETWEEN :lo AND :hi"""
        failures_sql = f"""WITH {_VALID_ORGS}, {_EVALS}, f AS (
                SELECT id, ts, org_uuid, kind, error FROM evals WHERE status = 'failed')"""
        failures = q(
            f"""{failures_sql}
                SELECT COUNT(*) OVER () AS total, f.id, {_ISO.format(col="f.ts")} AS created_at, f.org_uuid,
                       v.name AS org_name, f.kind, f.error
                  FROM f JOIN valid_orgs v ON v.uuid = f.org_uuid
                 WHERE {in_range}
                 ORDER BY datetime(f.ts) DESC LIMIT {LIST_LIMIT}"""
        )

        hits_sql = f"""WITH {_VALID_ORGS}, h AS (
                SELECT created_at AS ts, org_uuid, limit_name, kind, requested, cap FROM limit_hits
                UNION ALL SELECT created_at, org_uuid, 'max_scored_traces', 'trace_scoring', NULL, NULL
                            FROM ({_SCORING_CAP_HITS}))"""
        hits = q(
            f"""{hits_sql}
                SELECT COUNT(*) OVER () AS total, {_ISO.format(col="h.ts")} AS created_at, h.org_uuid, v.name AS org_name,
                       h.limit_name, h.kind, h.requested, h.cap
                  FROM h JOIN valid_orgs v ON v.uuid = h.org_uuid
                 WHERE {in_range}
                 ORDER BY datetime(h.ts) DESC LIMIT {LIST_LIMIT}"""
        )

    rows = [{**r, "metric": EVAL_FIELDS.get(r["metric"], r["metric"])} for r in map(dict, rows)]
    current_rows = [r for r in rows if lo <= r["day"] <= hi]

    # per_day: every day in range, zero-filled. All time starts at the first active day.
    days_seen = [r["day"] for r in current_rows] + [d for d in new_users_by_day if d >= lo]
    per_day = []
    if start or days_seen:
        day = start or date.fromisoformat(min(days_seen))
        by_day = defaultdict(lambda: {m: 0 for m in DAY_METRICS})
        active_by_day = defaultdict(set)
        for r in current_rows:
            if r["metric"] in DAY_METRICS:
                by_day[r["day"]][r["metric"]] += r["n"]
            if r["metric"] in ACTIVE_METRICS:
                active_by_day[r["day"]].add(r["org_uuid"])
        while day <= end:
            d = day.isoformat()
            per_day.append(
                {
                    "date": d,
                    "new_users": new_users_by_day.get(d, 0),
                    "active_workspaces": len(active_by_day[d]),
                    **by_day[d],
                }
            )
            day += timedelta(days=1)

    per_org = defaultdict(_empty_activity)
    for r in current_rows:
        _add_row(per_org[r["org_uuid"]], r)
    workspaces = []
    for o in orgs:
        a = per_org[o["uuid"]]
        workspaces.append(
            {
                "org_uuid": o["uuid"],
                "name": o["name"],
                "is_personal": bool(o["is_personal"]),
                "members": o["members"],
                "last_active_at": last_active.get(o["uuid"]),
                **{m: a[m] for m in RANGE_METRICS},
                "evaluations": sum(a[k] for k in EVAL_KINDS),
                "cost": a["cost"],
                "items_labelled": items_now.get(o["uuid"], 0),
            }
        )
    workspaces.sort(key=lambda w: w["last_active_at"] or "", reverse=True)

    return {
        "totals": {
            "users": users_total,
            "workspaces": len(orgs),
            "evaluations_running": running,
        },
        "activity": _summarise(rows, new_users_by_day, items_now, lo, hi),
        "previous": (
            _summarise(rows, new_users_by_day, items_prev, prev_lo, prev_hi) if start else None
        ),
        "per_day": per_day,
        "workspaces": workspaces,
        "failures": [_without_total(r) for r in failures],
        "failures_total": failures[0]["total"] if failures else 0,
        "limit_hits": [_without_total(r) for r in hits],
        "limit_hits_total": hits[0]["total"] if hits else 0,
    }


def _without_total(row) -> dict:
    out = dict(row)
    del out["total"]
    return out
