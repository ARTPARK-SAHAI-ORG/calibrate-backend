"""End-to-end proof that FAKE_AI_PROVIDERS wires the in-repo fake eval CLI into
the run → results pipeline.

Unlike the rest of the agent-test suite, this test does NOT patch
``subprocess.Popen``: it sets ``FAKE_AI_PROVIDERS=1`` and lets ``run_llm_test_task``
actually launch ``src/testing/fake_calibrate_agent.py`` and read its canned
output. Only the S3/queue side effects are stubbed. If the seam or the fake's
output contract regresses, the job won't reach ``done`` with ``passed == total``.
"""

import os
from unittest.mock import MagicMock, patch

import db


def _make_agent_with_response_test():
    user_uuid = db.create_user("F", "AI", f"fai-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    agent_uuid = db.create_agent(
        name=f"a-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )

    ev_uuid = db.create_evaluator(
        name=f"acc-{os.urandom(4).hex()}",
        evaluator_type="llm",
        output_type="binary",
        owner_user_id=user_uuid,
        org_uuid=org_uuid,
    )
    version = db.create_evaluator_version(
        ev_uuid, judge_model="m", system_prompt="judge this"
    )
    db.set_evaluator_live_version(ev_uuid, version["uuid"])

    test_uuid = db.create_test(
        name=f"t-{os.urandom(4).hex()}",
        type="response",
        config={
            "history": [{"role": "user", "content": "hi"}],
            "evaluation": {"type": "response"},
        },
        org_uuid=org_uuid,
        user_id=user_uuid,
    )
    db.set_test_evaluators(
        test_uuid, [{"evaluator_id": ev_uuid, "variable_values": None}]
    )

    job_uuid = db.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-unit-test", status="in_progress"
    )
    return db.get_agent(agent_uuid), db.get_test(test_uuid), job_uuid


def test_run_llm_test_task_end_to_end_with_fake_cli():
    from routers.agent_tests import run_llm_test_task

    agent, test, job_uuid = _make_agent_with_response_test()

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.agent_tests.get_s3_client", return_value=MagicMock()
    ), patch("routers.agent_tests.upload_directory_tree_to_s3"), patch(
        "routers.agent_tests.upload_file_to_s3"
    ), patch(
        "routers.agent_tests.try_start_queued_agent_test_job"
    ), patch(
        "routers.agent_tests.time.sleep"
    ):
        run_llm_test_task(job_uuid, agent, [test], "bucket")

    job = db.get_agent_test_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    results = job["results"]
    assert results["total_tests"] == 1
    assert results["passed"] == results["total_tests"]
    assert results["failed"] == 0
    assert results["test_results"][0]["passed"] is True
