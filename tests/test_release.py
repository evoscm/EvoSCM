import json
from unittest.mock import patch

import pytest

from scienceagent.release import WORLD_VARIANCES, require_world, run_session
from scienceagent.worlds import WORLDS, get_world
from scienceagent.transfer import _freeze_scm, _assert_no_forbidden_keys


def test_world_boundary():
    assert set(WORLDS) == set(WORLD_VARIANCES)
    assert len(WORLDS) == 11
    with pytest.raises(ValueError):
        get_world("unsupported", engine="nbody")


def test_export_excludes_evaluation_and_executable():
    source = {
        "scm_pipeline": {
            "candidates": [
                {"id": "H", "mechanisms": [], "parameters": {}, "final_law": "secret"}
            ],
            "posterior": {"H": 1.0},
            "final_claim": {"selected_candidate_id": "H"},
            "evidence": ["hidden"],
        },
        "evaluation": {"reference": "hidden"},
        "final_law": "hidden",
    }
    frozen = _freeze_scm(source, "source.json")
    _assert_no_forbidden_keys(frozen)
    assert "hidden" not in json.dumps(frozen)
    assert "secret" not in json.dumps(frozen)


@pytest.mark.parametrize("method", ["baseline", "evoscm"])
def test_real_session_with_mock_model(method):
    reply = "<final_law>\ndef discovered_law(pos1,pos2,p1,p2,velocity2,duration,**params):\n    import numpy as np\n    return np.asarray(pos2)+np.asarray(velocity2)*duration,np.asarray(velocity2)\n</final_law><explanation>Test law</explanation>"
    with patch("scienceagent.llm_client.complete", return_value=reply), patch(
        "scienceagent.release.evaluate_law",
        return_value={"mean_pos_error": 1.0, "explanation": {"score": 0.0}},
    ):
        state = run_session("gravity", "mock", method=method, max_rounds=1)
    assert state["world"] == "gravity"
    assert state["final_law"]
    assert state["evaluation"] is not None
    assert state["judge_model"] == "claude-opus-4-6"


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/"])
def test_claude_uses_separate_credentials(monkeypatch, suffix):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from scienceagent import llm_client

    monkeypatch.setenv("CLAUDE_API_KEY", "test-claude-key")
    monkeypatch.setenv("CLAUDE_BASE_URL", "https://claude.example.test" + suffix)
    monkeypatch.setenv("EVOSCM_API_TRANSPORT", "responses")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sdk = MagicMock()
    client = sdk.Anthropic.return_value.__enter__.return_value
    client.messages.create.return_value.content = [
        SimpleNamespace(type="text", text="<score>9</score>")
    ]
    with patch.dict("sys.modules", {"anthropic": sdk}):
        reply = llm_client.complete(
            "claude-opus-4-6",
            [{"role": "user", "content": "test"}],
            system="judge",
            max_tokens=1024,
        )
    assert reply == "<score>9</score>"
    assert sdk.Anthropic.call_args.kwargs["api_key"] == "test-claude-key"
    assert sdk.Anthropic.call_args.kwargs["base_url"] == "https://claude.example.test"
    assert client.messages.create.call_args.kwargs["system"] == "judge"


def test_transfer_compilation_protocol():
    from scienceagent.transfer import _build_prompt, _parse_response

    prompt = _build_prompt(
        {"mission": "test", "instructions": "test", "law_stub": "test"}, {}
    )
    assert "<frozen_scm>" in prompt
    law, explanation = _parse_response(
        "<final_law>def discovered_law(*args):\n    return 0</final_law><explanation>test</explanation>"
    )
    assert explanation == "test"
    assert "discovered_law" in law


def test_metrics_include_failed_sessions(tmp_path):
    from evaluate import summarize

    paths = []
    for seed in range(5):
        path = tmp_path / f"gravity_seed{seed}.json"
        result = (
            {"mean_pos_error": 0.01, "explanation": {"score": 1.0}}
            if seed == 0
            else None
        )
        path.write_text(
            json.dumps(
                {
                    "world": "gravity",
                    "noise_seed": seed,
                    "evaluation": result,
                    "simulator_episodes_used": 2,
                }
            )
        )
        paths.append(path)
    summary = summarize(paths)
    assert summary["pass_at_k"]["1"] == pytest.approx(0.2)
    assert summary["pass_at_k"]["5"] == 1.0
    assert summary["finite_mse_count"] == 1
    assert summary["nonfinite_mse_count"] == 4
    assert summary["missing_evaluation_count"] == 4


def test_judge_failure_is_not_reported_as_completed():
    reply = "<final_law>def discovered_law(*args):\n    return 0</final_law><explanation>test</explanation>"
    with patch("scienceagent.llm_client.complete", return_value=reply), patch(
        "scienceagent.release.evaluate_law",
        return_value={
            "mean_pos_error": 1.0,
            "explanation": {"score": None, "error": "request failed"},
        },
    ):
        row = run_session(
            "gravity", "mock", method="baseline", max_rounds=1, max_episodes=0
        )
    assert row["evaluation_error"] == "ExplanationEvaluationFailed"
    assert row["simulator_episodes_used"] == 0


def test_batch_validates_all_controls_before_running(tmp_path):
    from run_benchmark import prepare_sessions

    config = {
        "method": "evoscm",
        "model": "mock",
        "worlds": ["gravity"],
        "seeds": [0, 1],
    }
    control = {
        "world": "gravity",
        "noise_seed": 0,
        "method": "baseline",
        "engine": "nbody",
        "model": "mock",
        "judge_model": "claude-opus-4-6",
        "noise_frac": 0.05,
        "max_rounds": 1,
        "max_tokens": 100,
        "context_max_chars": None,
        "simulator_episodes_used": 0,
    }
    (tmp_path / "gravity_seed0.json").write_text(json.dumps(control))
    with pytest.raises(FileNotFoundError):
        prepare_sessions(config, tmp_path / "out", tmp_path)
    config["seeds"] = [0]
    plan = prepare_sessions(config, tmp_path / "out", tmp_path)
    assert plan[0][3]["max_episodes"] == 0
    control["method"] = "evoscm"
    (tmp_path / "gravity_seed0.json").write_text(json.dumps(control))
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_sessions(config, tmp_path / "out", tmp_path)


def test_no_extension_placeholders():
    from scienceagent.scm_pipeline import SCMPipeline

    pipeline = SCMPipeline(world="gravity", mission="test")
    assert not hasattr(pipeline, "replicated_population_audit")
    assert "replicated_affine_audit" not in pipeline.to_dict()
