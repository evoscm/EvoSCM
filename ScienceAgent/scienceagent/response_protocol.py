import re

_TOOL_OUTPUT = re.compile(
    r"<\s*/?\s*(?:experiment_output|mse_fit_output)\b", re.IGNORECASE
)
_ACTION = re.compile(
    r"<\s*(run_experiment|run_mse_fit|design_experiments|final_law)\s*>",
    re.IGNORECASE,
)


def validate_agent_reply(reply):
    match = _TOOL_OUTPUT.search(reply)
    if match is None:
        return reply, None
    prefix = reply[: match.start()].rstrip()
    actions = list(_ACTION.finditer(prefix))
    names = [action.group(1).lower() for action in actions]
    if not actions or len(names) != len(set(names)):
        raise ValueError("Untrusted tool output in model reply without a safe action")
    if "final_law" in names and len(names) != 1:
        raise ValueError("Conflicting actions before untrusted tool output")
    if "run_experiment" in names and "design_experiments" in names:
        raise ValueError("Conflicting experiment actions before untrusted tool output")
    for action in actions:
        closing = re.search(
            r"</\s*" + action.group(1) + r"\s*>",
            prefix[action.end() :],
            re.IGNORECASE,
        )
        if closing is None:
            raise ValueError("Incomplete action before untrusted tool output")
    return prefix, {
        "reason": "assistant_generated_tool_output",
        "original_chars": len(reply),
        "accepted_chars": len(prefix),
        "discarded_chars": len(reply) - len(prefix),
    }
