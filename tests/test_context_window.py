import copy
import json
from scienceagent.context_window import context_char_count, prepare_messages_for_context


def _trajectory(time_count=10, entity_count=35):
    times = [float(index) for index in range(time_count)]
    positions = [
        [
            [float(time_index + entity_index), float(entity_index)]
            for entity_index in range(entity_count)
        ]
        for time_index in range(time_count)
    ]
    velocities = [
        [
            [float(time_index), float(-entity_index)]
            for entity_index in range(entity_count)
        ]
        for time_index in range(time_count)
    ]
    return {
        "measurement_times": times,
        "positions": positions,
        "velocities": velocities,
        "background_initial_positions": positions[0][:-5],
    }


def _output_message(trajectories=1):
    payload = [_trajectory() for _ in range(trajectories)]
    return {
        "role": "user",
        "content": "<experiment_output>\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n</experiment_output>",
    }


def test_under_limit_is_an_equal_non_mutating_copy():
    messages = [{"role": "user", "content": "mission"}]
    original = copy.deepcopy(messages)
    prepared, report = prepare_messages_for_context(
        messages, system="system", max_chars=1000
    )
    assert prepared == original
    assert prepared is not messages
    assert messages == original
    assert report["observation_blocks_compacted"] == 0
    assert report["prepared_chars"] == report["raw_chars"]


def test_large_trajectory_gets_aligned_time_and_entity_context_view():
    messages = [{"role": "user", "content": "mission"}, _output_message(trajectories=2)]
    original = copy.deepcopy(messages)
    prepared, report = prepare_messages_for_context(
        messages, system="system", max_chars=12000
    )
    assert messages == original
    assert report["observation_blocks_compacted"] == 1
    assert report["prepared_chars"] <= 12000
    body = prepared[-1]["content"].split("\n", 1)[1].rsplit("\n", 1)[0]
    trajectory = json.loads(body)[0]
    assert trajectory["measurement_times"] == [0.0, 4.0, 9.0]
    assert len(trajectory["positions"]) == 3
    assert len(trajectory["positions"][0]) == 12
    assert len(trajectory["velocities"]) == 3
    assert len(trajectory["background_initial_positions"]) == 12
    assert trajectory["_context_view"]["full_trajectory_preserved_in_run_artifact"]
    assert trajectory["_context_view"]["original_entity_count"] == 35
    assert trajectory["_context_full_entity_stats"]["positions"][0]["count"] == 35


def test_compaction_is_deterministic_and_obeys_total_cap_with_system():
    messages = [{"role": "user", "content": "mission"}]
    for index in range(20):
        messages.append(
            {"role": "assistant", "content": f"analysis-{index}-" + "x" * 900}
        )
        messages.append(_output_message())
    first, first_report = prepare_messages_for_context(
        messages, system="s" * 2000, max_chars=20000
    )
    second, second_report = prepare_messages_for_context(
        messages, system="s" * 2000, max_chars=20000
    )
    assert first == second
    assert first_report == second_report
    assert context_char_count(first, system="s" * 2000) <= 20000
    assert first_report["messages_omitted"] > 0
    assert first_report["omitted_sha256"]
    assert first[0]["content"] == "mission"
    assert any(("<context_compaction>" in item["content"] for item in first))
