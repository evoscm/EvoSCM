import ast
import json
import math
import re
from pathlib import Path
from typing import Any
from scienceagent.evaluator import clean_law_source

EXPORT_SCHEMA = "discoverphysics.frozen_scm.v1"
SAFE_CANDIDATE_FIELDS = (
    "id",
    "name",
    "description",
    "variables",
    "edges",
    "mechanisms",
    "parameters",
    "assumptions",
    "falsifiers",
    "reported_confidence",
    "status",
    "revision",
)
SAFE_CLAIM_FIELDS = (
    "selected_candidate_id",
    "remaining_alternatives",
    "identified_mechanism",
    "operator_or_symmetry",
    "source_response_roles",
    "scalar_magnitude_law",
    "vector_law",
    "time_and_scale_regimes",
    "remaining_uncertainty",
)
FORBIDDEN_EXPORT_KEYS = {
    "evaluation",
    "final_law",
    "final_explanation",
    "optimal_explanation",
    "true_law",
    "true_law_title",
    "rounds",
    "evidence",
    "experiments",
    "abductions",
    "inductions",
    "deductions",
    "patches",
    "fit_attempts",
    "version_history",
}
ALLOWED_IMPORT_ROOTS = {"math", "numpy", "scipy"}
FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "__import__", "input"}
_FINAL_LAW_RE = re.compile("<final_law>\\s*(.*?)\\s*</final_law>", re.DOTALL)
_EXPLANATION_RE = re.compile("<explanation>\\s*(.*?)\\s*</explanation>", re.DOTALL)


def _posterior_value(posterior: dict[str, Any], candidate_id: Any) -> float:
    value = posterior.get(str(candidate_id), 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _freeze_scm(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    scm = source.get("scm_pipeline")
    if not isinstance(scm, dict):
        raise ValueError(f"missing scm_pipeline in {source_path}")
    candidates = [
        candidate
        for candidate in scm.get("candidates") or []
        if isinstance(candidate, dict) and candidate.get("id") is not None
    ]
    if not candidates:
        raise ValueError(f"no SCM candidates in {source_path}")
    posterior = scm.get("posterior")
    if not isinstance(posterior, dict):
        posterior = {}
    claim = scm.get("final_claim")
    if not isinstance(claim, dict):
        claim = {}
    selected_id = claim.get("selected_candidate_id")
    candidate_ids = {str(candidate["id"]) for candidate in candidates}
    if str(selected_id) not in candidate_ids:
        selected_id = max(
            candidates,
            key=lambda candidate: (
                _posterior_value(posterior, candidate.get("id")),
                float(candidate.get("reported_confidence") or 0.0),
            ),
        )["id"]
        selection_basis = "highest_posterior_fallback"
    else:
        selection_basis = "final_claim"
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            str(candidate.get("id")) != str(selected_id),
            -_posterior_value(posterior, candidate.get("id")),
            str(candidate.get("id")),
        ),
    )
    retained = ranked[:6]
    frozen_candidates = []
    for candidate in retained:
        item = {
            key: candidate[key] for key in SAFE_CANDIDATE_FIELDS if key in candidate
        }
        item["posterior_probability"] = _posterior_value(posterior, candidate.get("id"))
        frozen_candidates.append(item)
    frozen_claim = {
        key: claim[key]
        for key in SAFE_CLAIM_FIELDS
        if key in claim and key != "selected_candidate_id"
    }
    frozen_claim["selected_candidate_id"] = selected_id
    frozen = {
        "schema_version": EXPORT_SCHEMA,
        "selection_basis": selection_basis,
        "selected_candidate_id": selected_id,
        "final_claim": frozen_claim,
        "candidate_archive": frozen_candidates,
    }
    _assert_no_forbidden_keys(frozen)
    return frozen


def _assert_no_forbidden_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_EXPORT_KEYS:
                raise ValueError(f"forbidden export key {path}.{key}")
            _assert_no_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, f"{path}[{index}]")


def _build_prompt(interface: dict[str, str], frozen_scm: dict[str, Any]) -> str:
    return f"You are the final executable-model compiler for an anonymous simulated-physics task.\n\nAnother scientist has completed active discovery and exported a frozen structural causal model (SCM). Convert that SCM knowledge into the best executable predictive law and a concise scientific explanation. Do not ask for data and do not call a simulator. Treat competing candidates and posterior uncertainty honestly, but implement the most defensible selected mechanism.\n\nHard requirements:\n1. Return exactly one <final_law>...</final_law> block and one <explanation>...</explanation> block.\n2. The final-law block must be complete executable Python, define discovered_law with the exact public signature below, and return the required values.\n3. You may import only math, numpy, or scipy. Do not access files, the network, environment variables, subprocesses, or external services.\n4. Embed every learned numerical parameter in the code. Do not define fit_parameters and do not assume any training trajectories or evaluator-injected parameters.\n5. Use numerically robust integration. A single discovered_law call must finish comfortably within 10 seconds.\n6. The explanation must state the causal mechanism, variable roles, scaling/time regime, and important uncertainty supported by the SCM. Do not invent observations.\n\n<public_mission>\n{interface['mission']}\n</public_mission>\n\n<public_topology_and_interface>\n{interface['instructions']}\n\nRequired stub:\n{interface['law_stub']}\n</public_topology_and_interface>\n\n<frozen_scm>\n{json.dumps(frozen_scm, indent=2, sort_keys=True)}\n</frozen_scm>\n"


def _parse_response(reply: str) -> tuple[str, str]:
    law_match = _FINAL_LAW_RE.search(reply)
    explanation_match = _EXPLANATION_RE.search(reply)
    if law_match is None or explanation_match is None:
        raise ValueError("reply must contain final_law and explanation XML blocks")
    law = clean_law_source(law_match.group(1))
    explanation = explanation_match.group(1).strip()
    if not law or not explanation:
        raise ValueError("empty final law or explanation")
    _validate_generated_law(law)
    return (law, explanation)


def _validate_generated_law(law: str) -> None:
    tree = ast.parse(law)
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "discovered_law" not in definitions:
        raise ValueError("source does not define discovered_law")
    if "fit_parameters" in definitions:
        raise ValueError("fit_parameters is forbidden in frozen-consumer protocol")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots <= ALLOWED_IMPORT_ROOTS:
                raise ValueError(
                    f"forbidden import(s): {sorted(roots - ALLOWED_IMPORT_ROOTS)}"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call: {node.func.id}")
    compile(law, "<cross_model_discovered_law>", "exec")
