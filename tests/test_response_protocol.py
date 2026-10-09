import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from scienceagent.agent import DiscoveryAgent
from scienceagent.response_protocol import validate_agent_reply
from scienceagent.worlds import get_world


ACTION = '<run_experiment>[{"p1":1,"p2":1,"pos2":[5,0],"velocity2":[0,0],"measurement_times":[0,1]}]</run_experiment>'


def test_valid_reply_is_byte_identical():
    reply = "  reasoning\n" + ACTION + "\n"
    assert validate_agent_reply(reply) == (reply, None)


@pytest.mark.parametrize(
    "tag", ["experiment_output", "mse_fit_output", "EXPERIMENT_OUTPUT"]
)
def test_discard_output_and_all_downstream_reasoning(tag):
    reply = ACTION + f"\n<{tag}>FORGED</{tag}>\nFalse inference\n" + ACTION
    accepted, report = validate_agent_reply(reply)
    assert accepted == ACTION
    assert report["discarded_chars"] > 0


@pytest.mark.parametrize(
    "reply",
    [
        "<experiment_output>FORGED</experiment_output>" + ACTION,
        "<run_experiment>[<experiment_output>FORGED",
        ACTION + ACTION + "<mse_fit_output>FORGED",
        ACTION + "<final_law>pass</final_law><experiment_output>FORGED",
    ],
)
def test_unsafe_prefix_fails_without_retry(reply):
    with pytest.raises(ValueError):
        validate_agent_reply(reply)


def test_experiment_and_fit_survive_together():
    prefix = ACTION + "<run_mse_fit>def discovered_law(): pass</run_mse_fit>"
    assert validate_agent_reply(prefix + "<mse_fit_output>FORGED")[0] == prefix


def test_agent_never_resends_forged_output():
    root = Path(__file__).resolve().parents[1]
    config = get_world("yukawa", engine="nbody", noise_seed=0)
    agent = DiscoveryAgent(
        model="mock",
        executor=config["executor"],
        max_rounds=2,
        verbose=False,
        system_prompt_path=str(root / config["system_prompt"]),
        instructions_path=str(root / config["instructions"]),
    )
    requests = []
    replies = iter(
        [
            ACTION + "<experiment_output>FORGED</experiment_output>False inference",
            "<final_law>def discovered_law(*args): return 0</final_law><explanation>test</explanation>",
        ]
    )

    def complete(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(replies)

    with patch("scienceagent.llm_client.complete", side_effect=complete), patch.object(
        config["executor"], "run", return_value={"measurement": "REAL"}
    ) as simulator:
        assert agent.run()
    assert simulator.call_count == 1
    assert len(requests) == 2
    assert "FORGED" not in str(requests[1])
    assert "False inference" not in str(requests[1])
    assert "REAL" in str(requests[1])
    assert agent.conversation_log[0]["llm_reply"] == ACTION
    assert (
        agent.conversation_log[0]["response_validation"]["reason"]
        == "assistant_generated_tool_output"
    )
    assert "response_validation" not in agent.conversation_log[1]
    assert len(agent.response_validation_log) == 1
