from __future__ import annotations
import hashlib
import json
import math
import re
from typing import Any, Optional

DEFAULT_CONTEXT_TIME_POINTS = 3
DEFAULT_CONTEXT_ENTITIES = 12
_OUTPUT_BLOCK_RE = re.compile(
    "<(?P<tag>experiment_output|counterfactual_output)>\\s*(?P<body>.*?)\\s*</(?P=tag)>",
    re.DOTALL,
)
_TIME_ALIGNED_FIELDS = (
    "positions",
    "pos1",
    "pos2",
    "velocities",
    "velocity1",
    "velocity2",
)
_ENTITY_ALIGNED_FIELDS = ("particle_charges", "particle_masses")
_BACKGROUND_FIELDS = ("background_initial_positions", "background_initial_velocities")


def prepare_messages_for_context(
    messages: list[dict],
    *,
    system: Optional[str],
    max_chars: Optional[int],
    max_time_points: int = DEFAULT_CONTEXT_TIME_POINTS,
    max_entities: int = DEFAULT_CONTEXT_ENTITIES,
) -> tuple[list[dict], dict]:
    copied = [_copy_message(message) for message in messages]
    raw_chars = context_char_count(copied, system=system)
    report = {
        "enabled": max_chars is not None,
        "max_chars": max_chars,
        "raw_chars": raw_chars,
        "prepared_chars": raw_chars,
        "observation_blocks_compacted": 0,
        "messages_omitted": 0,
        "omitted_sha256": None,
        "hard_truncations": 0,
        "max_time_points": max_time_points,
        "max_entities": max_entities,
    }
    if max_chars is None or raw_chars <= max_chars:
        return (copied, report)
    if max_chars < 1:
        raise ValueError("max_chars must be a positive integer or None")
    if max_time_points < 2:
        raise ValueError("max_time_points must be at least 2")
    if max_entities < 2:
        raise ValueError("max_entities must be at least 2")
    compacted = []
    compacted_blocks = 0
    for message in copied:
        content, block_count = _compact_output_blocks(
            message["content"],
            max_time_points=max_time_points,
            max_entities=max_entities,
        )
        compacted.append({**message, "content": content})
        compacted_blocks += block_count
    report["observation_blocks_compacted"] = compacted_blocks
    if context_char_count(compacted, system=system) > max_chars:
        compacted, tail_report = _retain_mission_and_recent_tail(
            compacted, system=system, max_chars=max_chars
        )
        report.update(tail_report)
    prepared_chars = context_char_count(compacted, system=system)
    if prepared_chars > max_chars:
        raise RuntimeError(
            f"context preparation failed to satisfy its deterministic limit: {prepared_chars} > {max_chars}"
        )
    report["prepared_chars"] = prepared_chars
    return (compacted, report)


def context_char_count(messages: list[dict], *, system: Optional[str]) -> int:
    return len(system or "") + sum(
        (len(str(message.get("content", ""))) + 16 for message in messages)
    )


def _copy_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise TypeError("each message must be a mapping")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise TypeError("message content must be a string")
    return {**message, "content": content}


def _compact_output_blocks(
    content: str, *, max_time_points: int, max_entities: int
) -> tuple[str, int]:
    compacted_count = 0

    def replace(match: re.Match) -> str:
        nonlocal compacted_count
        try:
            parsed = json.loads(match.group("body"))
        except json.JSONDecodeError:
            return match.group(0)
        view, changed = _context_view(
            parsed, max_time_points=max_time_points, max_entities=max_entities
        )
        if not changed:
            return match.group(0)
        compacted_count += 1
        tag = match.group("tag")
        body = json.dumps(view, separators=(",", ":"), ensure_ascii=False)
        return f"<{tag}>\n{body}\n</{tag}>"

    return (_OUTPUT_BLOCK_RE.sub(replace, content), compacted_count)


def _context_view(
    value: Any, *, max_time_points: int, max_entities: int
) -> tuple[Any, bool]:
    if isinstance(value, list):
        changed = False
        result = []
        for item in value:
            compacted, item_changed = _context_view(
                item, max_time_points=max_time_points, max_entities=max_entities
            )
            result.append(compacted)
            changed = changed or item_changed
        return (result, changed)
    if not isinstance(value, dict):
        return (value, False)
    times = value.get("measurement_times")
    if isinstance(times, list) and len(times) > max_time_points:
        return _compact_trajectory_dict(
            value, max_time_points=max_time_points, max_entities=max_entities
        )
    changed = False
    result = {}
    for key, item in value.items():
        compacted, item_changed = _context_view(
            item, max_time_points=max_time_points, max_entities=max_entities
        )
        result[key] = compacted
        changed = changed or item_changed
    return (result, changed)


def _compact_trajectory_dict(
    value: dict, *, max_time_points: int, max_entities: int
) -> tuple[dict, bool]:
    times = list(value["measurement_times"])
    time_indices = _even_indices(len(times), max_time_points)
    entity_count = _infer_entity_count(value, len(times))
    entity_indices = (
        _even_indices(entity_count, max_entities)
        if entity_count > max_entities
        else list(range(entity_count))
    )
    result: dict[str, Any] = {}
    full_entity_stats: dict[str, list[dict]] = {}
    for key, item in value.items():
        if key == "measurement_times":
            result[key] = [times[index] for index in time_indices]
            continue
        if key in _TIME_ALIGNED_FIELDS and _is_time_aligned(item, len(times)):
            selected_rows = [item[index] for index in time_indices]
            if entity_count > max_entities and _has_entity_axis(item, entity_count):
                stats = [_vector_stats(row) for row in selected_rows]
                if all((stat is not None for stat in stats)):
                    full_entity_stats[key] = stats
                selected_rows = [
                    [row[index] for index in entity_indices] for row in selected_rows
                ]
            result[key] = selected_rows
            continue
        if (
            key in _ENTITY_ALIGNED_FIELDS
            and entity_count > max_entities
            and isinstance(item, list)
            and (len(item) == entity_count)
        ):
            result[key] = [item[index] for index in entity_indices]
            continue
        if key in _BACKGROUND_FIELDS and isinstance(item, list):
            result[key], background_meta = _compact_entity_rows(
                item, max_entities=max_entities
            )
            if background_meta is not None:
                result.setdefault("_context_background_stats", {})[
                    key
                ] = background_meta
            continue
        compacted, _ = _context_view(
            item, max_time_points=max_time_points, max_entities=max_entities
        )
        result[key] = compacted
    if full_entity_stats:
        result["_context_full_entity_stats"] = full_entity_stats
    result["_context_view"] = {
        "full_trajectory_preserved_in_run_artifact": True,
        "original_measurement_times": times,
        "selected_measurement_indices": time_indices,
        "original_entity_count": entity_count,
        "selected_entity_indices": entity_indices,
    }
    return (result, True)


def _infer_entity_count(value: dict, time_count: int) -> int:
    for key in ("positions", "velocities"):
        item = value.get(key)
        if _is_time_aligned(item, time_count) and item and isinstance(item[0], list):
            if item[0] and isinstance(item[0][0], list):
                return len(item[0])
    return 0


def _is_time_aligned(value: Any, time_count: int) -> bool:
    return isinstance(value, list) and len(value) == time_count


def _has_entity_axis(value: list, entity_count: int) -> bool:
    return bool(
        value
        and isinstance(value[0], list)
        and (len(value[0]) == entity_count)
        and value[0]
        and isinstance(value[0][0], list)
    )


def _compact_entity_rows(
    rows: list, *, max_entities: int
) -> tuple[list, Optional[dict]]:
    if len(rows) <= max_entities:
        return (rows, None)
    indices = _even_indices(len(rows), max_entities)
    return (
        [rows[index] for index in indices],
        {
            "original_entity_count": len(rows),
            "selected_entity_indices": indices,
            "full_population_stats": _vector_stats(rows),
        },
    )


def _vector_stats(rows: list) -> Optional[dict]:
    if not rows or not all((isinstance(row, list) and row for row in rows)):
        return None
    width = len(rows[0])
    if not all(
        (
            len(row) == width
            and all((isinstance(value, (int, float)) for value in row))
            for row in rows
        )
    ):
        return None
    columns = [[float(row[index]) for row in rows] for index in range(width)]
    return {
        "count": len(rows),
        "mean": [_round(sum(column) / len(column)) for column in columns],
        "min": [_round(min(column)) for column in columns],
        "max": [_round(max(column)) for column in columns],
        "rms": [
            _round(math.sqrt(sum((value * value for value in column)) / len(column)))
            for column in columns
        ],
    }


def _round(value: float) -> float:
    return round(value, 4)


def _even_indices(length: int, maximum: int) -> list[int]:
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    return sorted(
        {int(round(index * (length - 1) / (maximum - 1))) for index in range(maximum)}
    )


def _retain_mission_and_recent_tail(
    messages: list[dict], *, system: Optional[str], max_chars: int
) -> tuple[list[dict], dict]:
    if not messages:
        return (
            [],
            {"messages_omitted": 0, "omitted_sha256": None, "hard_truncations": 0},
        )
    first = messages[0]
    tail_start = 1
    selected_tail = messages[tail_start:]
    hard_truncations = 0
    while selected_tail:
        omitted = messages[tail_start : len(messages) - len(selected_tail)]
        marker = _omission_marker(omitted) if omitted else None
        candidate = [first]
        if marker is not None:
            candidate.append(marker)
        candidate.extend(selected_tail)
        if context_char_count(candidate, system=system) <= max_chars:
            break
        if len(selected_tail) == 1:
            break
        selected_tail = selected_tail[1:]
    omitted = messages[tail_start : len(messages) - len(selected_tail)]
    marker = _omission_marker(omitted) if omitted else None
    prepared = [first]
    if marker is not None:
        prepared.append(marker)
    prepared.extend(selected_tail)
    while context_char_count(prepared, system=system) > max_chars and prepared:
        mutable_indices = list(range(len(prepared)))
        largest_index = max(
            mutable_indices, key=lambda index: len(prepared[index].get("content", ""))
        )
        excess = context_char_count(prepared, system=system) - max_chars
        old = prepared[largest_index]["content"]
        target = max(0, len(old) - excess - 256)
        prepared[largest_index] = {
            **prepared[largest_index],
            "content": _truncate_with_hash(old, target),
        }
        hard_truncations += 1
        if not old:
            break
    digest = _messages_digest(omitted) if omitted else None
    return (
        prepared,
        {
            "messages_omitted": len(omitted),
            "omitted_sha256": digest,
            "hard_truncations": hard_truncations,
        },
    )


def _omission_marker(messages: list[dict]) -> dict:
    digest = _messages_digest(messages)
    return {
        "role": "user",
        "content": f"<context_compaction>omitted_messages={len(messages)};sha256={digest};policy=deterministic_recent_tail;authoritative_full_history_preserved_in_run_artifact</context_compaction>",
    }


def _messages_digest(messages: list[dict]) -> str:
    payload = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truncate_with_hash(content: str, target: int) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    marker = f"<hard_context_truncation sha256={digest}>"
    if target <= len(marker):
        return marker[:target]
    remaining = target - len(marker)
    head = remaining // 2
    tail = remaining - head
    return content[:head] + marker + (content[-tail:] if tail else "")
