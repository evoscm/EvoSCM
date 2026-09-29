import json
from unittest.mock import patch
import numpy as np
import pytest
from scipy.special import k1 as bessel_k1
from scienceagent.agent import DiscoveryAgent
from scienceagent.evaluator import _extract_training_trajectories
from scienceagent.scm_pipeline import (
    SCMPipeline,
    resolve_json_pointer,
    set_json_pointer,
)


def _candidate(candidate_id, confidence=0.5, exponent=1.0):
    return {
        "id": candidate_id,
        "name": candidate_id,
        "description": "radial dynamic SCM",
        "confidence": confidence,
        "variables": [
            {"name": "r", "role": "state", "observed": True},
            {"name": "q", "role": "parameter", "observed": False},
        ],
        "edges": [{"source": "r", "target": "a"}],
        "mechanisms": [
            {"target": "a", "parents": ["r"], "equation": f"a=-k/r**{exponent}"}
        ],
        "parameters": {"q": {"estimate": exponent, "lower": 0.0, "upper": 3.0}},
        "falsifiers": ["incorrect radial scaling"],
    }


def _update_block(*candidates):
    return (
        "<scm_update>"
        + json.dumps(
            {
                "mode": "initialize",
                "abduction": {"latent_causes": ["unknown radial exponent"]},
                "candidates": list(candidates),
                "inductions": [],
                "deductions": [{"claim": "radius changes response"}],
                "open_questions": ["which exponent?"],
            }
        )
        + "</scm_update>"
    )


def _design_block(designs):
    return (
        "<design_experiments>"
        + json.dumps({"designs": designs})
        + "</design_experiments>"
    )


def test_json_pointer_supports_negative_indices_and_safe_interventions():
    document = {"items": [{"trajectory": [[1.0, 2.0], [3.0, 4.0]]}]}
    assert resolve_json_pointer(document, "/items/0/trajectory/-1/0") == 3.0
    set_json_pointer(document, "/items/0/trajectory/0/1", 9.0)
    assert document["items"][0]["trajectory"][0] == [1.0, 9.0]
    with pytest.raises(KeyError):
        set_json_pointer(document, "/items/0/missing", 1.0)
    singleton_batch = [{"velocity": [[1.0, 2.0], [3.0, 4.0]]}]
    assert resolve_json_pointer(singleton_batch, "/velocity/-1/0") == 3.0


def test_agent_facing_runtime_state_does_not_leak_world_label():
    pipeline = SCMPipeline(world="fractional", strict=True)
    context = pipeline.round_context(round_num=1)
    assert '"world"' not in context
    assert "fractional" not in context
    assert pipeline.to_dict()["world"] == "fractional"


def test_semigroup_check_rejects_frozen_initial_position_dynamics():
    pipeline = SCMPipeline(world="opaque_internal_id", strict=True)
    pipeline.experiments = [
        {
            "kind": "intervention",
            "input": [
                {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [radius, 0.0],
                    "velocity2": [0.0, 0.0],
                    "measurement_times": [0.25, 0.5],
                }
                for radius in (2.0, 4.0, 8.0)
            ],
        }
    ]
    correct = "\ndef discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):\n    import math\n    c = math.cos(duration)\n    s = math.sin(duration)\n    x = [pos2[i] * c + velocity2[i] * s for i in range(2)]\n    v = [-pos2[i] * s + velocity2[i] * c for i in range(2)]\n    return x, v\n"
    frozen_initial_position = "\ndef discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):\n    acceleration = [-pos2[0], -pos2[1]]\n    x = [\n        pos2[i] + velocity2[i] * duration\n        + 0.5 * acceleration[i] * duration * duration\n        for i in range(2)\n    ]\n    v = [velocity2[i] + acceleration[i] * duration for i in range(2)]\n    return x, v\n"
    good = pipeline.executable_semigroup_check(correct)
    bad = pipeline.executable_semigroup_check(frozen_initial_position)
    assert good["passed"] is True
    assert good["max_relative_state_gap"] < 1e-10
    assert bad["passed"] is False
    assert bad["max_relative_state_gap"] > bad["relative_tolerance"]
    assert good["simulator_episodes_spent"] == 0


def test_runtime_state_reports_control_coverage_and_prompts_unresolved_regime():
    pipeline = SCMPipeline(world="opaque_internal_id", strict=True)
    pipeline.ingest_reply(
        _update_block(
            {
                **_candidate("H_crossover"),
                "description": "central law with a short-range crossover",
            },
            _candidate("H_power"),
        ),
        round_num=1,
    )
    pipeline.experiments = [
        {
            "kind": "counterfactual",
            "input": {
                "factual": {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                },
                "interventions": [{"id": "far", "set": {"/pos2/0": 4.0}}],
            },
        }
    ]
    summary = pipeline.state_summary()
    context = pipeline.round_context(round_num=2)
    assert summary["intervention_coverage"]["executed_cases"] == 2
    assert summary["intervention_coverage"]["pos2_radius"]["min"] == 2.0
    assert summary["intervention_coverage"]["pos2_radius"]["max"] == 4.0
    assert "four distinct radii spanning 8x" in context
    assert "below the current minimum (2" in context
    assert "sign-flip" not in context


def test_runtime_state_requests_null_source_to_identify_absolute_sign():
    pipeline = SCMPipeline(world="opaque_internal_id", strict=True)
    pipeline.ingest_reply(
        _update_block(
            {
                **_candidate("H_attraction"),
                "description": "static central attraction from the source",
            },
            {
                **_candidate("H_repulsion"),
                "description": "static central repulsion from the source",
            },
        ),
        round_num=1,
    )
    pipeline.experiments = [
        {
            "kind": "intervention",
            "input": [
                {"p1": 1.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]}
            ],
        }
    ]
    context = pipeline.round_context(round_num=2)
    assert "p1=0" in context
    assert "fixes the sign" in context


def test_runtime_state_requires_successful_fit_for_parametric_candidates():
    pipeline = SCMPipeline(world="opaque_internal_id", strict=True)
    pipeline.ingest_reply(
        _update_block(_candidate("H_power"), _candidate("H_screened")), round_num=1
    )
    pipeline.experiments = [
        {
            "kind": "intervention",
            "input": [
                {"p1": 0.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]}
            ],
        }
    ]
    before = pipeline.round_context(round_num=3)
    assert "<run_mse_fit>" in before
    pipeline.record_mse_fit(
        3,
        {
            "loss_before": 1.0,
            "loss_after": 0.1,
            "fitted_params": {"q": 1.2},
            "declared_params": {"q": {"init": 1.0, "bounds": [0.5, 2.0]}},
            "n_training": 3,
            "error": None,
        },
    )
    after = pipeline.round_context(round_num=4)
    assert "continuous parameters remain uncalibrated" not in after
    assert pipeline.state_summary()["latest_mse_fit"]["loss_after"] == 0.1
    assert pipeline.metrics()["successful_mse_fits"] == 1


def test_pipeline_versions_candidates_and_applies_structured_patch():
    pipeline = SCMPipeline("test", strict=True)
    result = pipeline.ingest_reply(
        _update_block(_candidate("H1"), _candidate("H2", exponent=2.0)), round_num=1
    )
    assert result["protocol_errors"] == []
    assert set(pipeline.posterior()) == {"H1", "H2"}
    patch_reply = (
        "<scm_update>"
        + json.dumps(
            {
                "mode": "revise",
                "abduction": {"latent_causes": ["a hidden object class"]},
                "patches": [
                    {
                        "candidate_id": "H1",
                        "operation": "add_latent",
                        "payload": {
                            "variable": {"name": "species", "role": "latent class"}
                        },
                        "reason": "residual clusters",
                    },
                    {
                        "candidate_id": "H1",
                        "operation": "update_parameter",
                        "payload": {
                            "name": "q",
                            "estimate": 1.2,
                            "lower": 0.8,
                            "upper": 1.5,
                        },
                        "reason": "fit to two radii",
                    },
                ],
            }
        )
        + "</scm_update>"
    )
    pipeline.ingest_reply(patch_reply, round_num=2)
    assert pipeline.candidates["H1"]["revision"] == 3
    assert pipeline.candidates["H1"]["parameters"]["q"]["estimate"] == 1.2
    latent = [
        variable
        for variable in pipeline.candidates["H1"]["variables"]
        if variable["name"] == "species"
    ]
    assert latent and latent[0]["observed"] is False
    assert len(pipeline.version_history) == 2
    assert pipeline.metrics()["patches_applied"] == 2


def test_initial_candidate_confidences_are_audited_but_prior_is_uniform():
    pipeline = SCMPipeline("test", complexity_penalty=0.0)
    pipeline.ingest_reply(
        _update_block(
            _candidate("familiar", confidence=0.9),
            _candidate("unfamiliar", confidence=0.1, exponent=2.0),
        ),
        round_num=1,
    )
    assert pipeline.posterior() == pytest.approx({"familiar": 0.5, "unfamiliar": 0.5})
    assert pipeline.candidates["familiar"]["reported_confidence"] == 0.9
    assert pipeline.candidates["unfamiliar"]["reported_confidence"] == 0.1


def test_active_design_selection_prefers_larger_model_disagreement():
    pipeline = SCMPipeline("test")
    reply = _update_block(_candidate("H1"), _candidate("H2", exponent=2.0))
    reply += _design_block(
        [
            {
                "id": "low_disagreement",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.5,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/0/y",
                        "mean": 1.1,
                        "sigma": 0.5,
                    },
                ],
            },
            {
                "id": "high_disagreement",
                "kind": "intervention",
                "experiment": {"x": 4.0},
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.2,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/0/y",
                        "mean": 5.0,
                        "sigma": 0.2,
                    },
                ],
            },
        ]
    )
    parsed = pipeline.ingest_reply(reply, round_num=1)
    assert parsed["selected_design"]["design_id"] == "high_disagreement"
    scores = {
        row["design_id"]: row["information_score"]
        for row in parsed["selected_design"]["proposal_scores"]
    }
    assert scores["high_disagreement"] > scores["low_disagreement"]


def test_active_design_selection_respects_remaining_episode_budget():
    pipeline = SCMPipeline("test")
    reply = _update_block(_candidate("H1"), _candidate("H2", exponent=2.0))
    reply += _design_block(
        [
            {
                "id": "expensive",
                "kind": "counterfactual",
                "factual": {"x": 1.0},
                "interventions": [
                    {"id": "x2", "set": {"/x": 2.0}},
                    {"id": "x3", "set": {"/x": 3.0}},
                ],
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/counterfactuals/0/output/0/y",
                        "mean": 2.0,
                        "sigma": 0.1,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/counterfactuals/0/output/0/y",
                        "mean": 20.0,
                        "sigma": 0.1,
                    },
                ],
            },
            {
                "id": "affordable",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.5,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/0/y",
                        "mean": 1.2,
                        "sigma": 0.5,
                    },
                ],
            },
        ]
    )
    parsed = pipeline.ingest_reply(reply, round_num=1, max_episode_cost=1)
    assert parsed["selected_design"]["design_id"] == "affordable"
    assert pipeline.pending_episode_cost() == 1
    scores = {
        row["design_id"]: row for row in parsed["selected_design"]["proposal_scores"]
    }
    assert scores["expensive"]["simulator_episode_cost"] == 3
    assert scores["expensive"]["budget_feasible"] is False
    assert scores["affordable"]["budget_feasible"] is True


def test_active_selector_prioritizes_missing_null_source_audit():
    pipeline = SCMPipeline("test", strict=True)
    pipeline.ingest_reply(
        _update_block(
            {
                **_candidate("H_attract"),
                "description": "central source-driven attraction",
            },
            {
                **_candidate("H_repulse"),
                "description": "central source-driven repulsion",
            },
        ),
        round_num=1,
    )
    pipeline.experiments = [
        {
            "kind": "intervention",
            "input": [
                {"p1": 1.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]}
            ],
        }
    ]
    reply = _design_block(
        [
            {
                "id": "high_score_without_null",
                "kind": "intervention",
                "experiment": {
                    "p1": 2.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                },
                "predictions": [
                    {
                        "hypothesis_id": "H_attract",
                        "target": "/0/y",
                        "mean": -10.0,
                        "sigma": 0.1,
                    },
                    {
                        "hypothesis_id": "H_repulse",
                        "target": "/0/y",
                        "mean": 10.0,
                        "sigma": 0.1,
                    },
                ],
            },
            {
                "id": "required_null_source",
                "kind": "counterfactual",
                "factual": {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                },
                "interventions": [{"id": "null", "set": {"/p1": 0.0}}],
                "predictions": [
                    {
                        "hypothesis_id": "H_attract",
                        "target": "/paired_effects/0/output_delta/0/y",
                        "mean": 0.1,
                        "sigma": 1.0,
                    },
                    {
                        "hypothesis_id": "H_repulse",
                        "target": "/paired_effects/0/output_delta/0/y",
                        "mean": -0.1,
                        "sigma": 1.0,
                    },
                ],
            },
        ]
    )
    parsed = pipeline.ingest_reply(reply, round_num=2)
    assert parsed["selected_design"]["design_id"] == "required_null_source"
    scores = {
        row["design_id"]: row for row in parsed["selected_design"]["proposal_scores"]
    }
    assert (
        scores["high_score_without_null"]["information_score"]
        > scores["required_null_source"]["information_score"]
    )
    assert scores["required_null_source"]["satisfies_null_source_audit"] is True


def test_active_selector_prioritizes_absolute_start_time_audit():
    pipeline = SCMPipeline("test", strict=True)
    pipeline.ingest_reply(
        _update_block(
            {**_candidate("H_static"), "description": "static radial response"},
            {**_candidate("H_phase"), "description": "another static radial response"},
        ),
        round_num=1,
    )
    pipeline.experiments = [
        {
            "kind": "intervention",
            "input": [
                {"p1": 0.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]}
            ],
        }
    ]
    reply = _design_block(
        [
            {
                "id": "high_score_without_clock",
                "kind": "intervention",
                "experiment": {
                    "p1": 0.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                },
                "predictions": [
                    {
                        "hypothesis_id": "H_static",
                        "target": "/0/y",
                        "mean": -10.0,
                        "sigma": 0.1,
                    },
                    {
                        "hypothesis_id": "H_phase",
                        "target": "/0/y",
                        "mean": 10.0,
                        "sigma": 0.1,
                    },
                ],
            },
            {
                "id": "required_clock_shift",
                "kind": "intervention",
                "experiments": [
                    {
                        "p1": 0.0,
                        "p2": 1.0,
                        "pos2": [2.0, 0.0],
                        "velocity2": [0.0, 0.0],
                        "start_time": 0.0,
                    },
                    {
                        "p1": 0.0,
                        "p2": 1.0,
                        "pos2": [2.0, 0.0],
                        "velocity2": [0.0, 0.0],
                        "start_time": 2.0,
                    },
                    {
                        "p1": 0.0,
                        "p2": 1.0,
                        "pos2": [2.0, 0.0],
                        "velocity2": [0.0, 0.0],
                        "start_time": 4.0,
                    },
                ],
                "predictions": [
                    {
                        "hypothesis_id": "H_static",
                        "target": "/0/y",
                        "mean": 0.0,
                        "sigma": 1.0,
                    },
                    {
                        "hypothesis_id": "H_phase",
                        "target": "/0/y",
                        "mean": 0.1,
                        "sigma": 1.0,
                    },
                ],
            },
        ]
    )
    parsed = pipeline.ingest_reply(reply, round_num=2)
    assert parsed["selected_design"]["design_id"] == "required_clock_shift"
    scores = {
        row["design_id"]: row for row in parsed["selected_design"]["proposal_scores"]
    }
    assert scores["required_clock_shift"]["satisfies_start_time_audit"] is True
    pipeline.experiments.append(
        {
            "kind": "intervention",
            "input": [
                {
                    "p1": 0.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "start_time": 2.0,
                }
            ],
        }
    )
    assert pipeline._needs_start_time_audit() is True
    pipeline.experiments.append(
        {
            "kind": "intervention",
            "input": [
                {
                    "p1": 0.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "start_time": 4.0,
                }
            ],
        }
    )
    assert pipeline._needs_start_time_audit() is False


def test_sign_audit_requires_and_selects_both_matched_scalar_reversals():
    pipeline = SCMPipeline("opaque_internal_id", strict=True, observation_noise_std=0.1)
    pipeline.ingest_reply(
        _update_block(_candidate("H_signed"), _candidate("H_magnitude")), round_num=1
    )
    pipeline.experiments = [
        {
            "kind": "counterfactual",
            "input": {
                "factual": {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                },
                "interventions": [{"id": "null", "set": {"/p1": 0.0}}],
            },
            "local_causal_edges": [
                _anchored_radial_edge(p1=1.0, p2=1.0, radius=2.0, inward=0.25)
            ],
        }
    ]
    assert pipeline._missing_sign_audits() == ["p1", "p2"]
    sign_design = {
        "id": "matched_signs",
        "kind": "counterfactual",
        "factual": {"p1": 1.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]},
        "interventions": [
            {"id": "negative_source", "set": {"/p1": -1.0}},
            {"id": "negative_probe", "set": {"/p2": -1.0}},
        ],
        "predictions": [
            {
                "hypothesis_id": hypothesis_id,
                "target": "/paired_effects/0/output_delta/0/y",
                "mean": mean,
                "sigma": 1.0,
            }
            for hypothesis_id, mean in (("H_signed", -0.1), ("H_magnitude", 0.1))
        ],
    }
    high_disagreement = {
        "id": "high_disagreement_without_signs",
        "kind": "intervention",
        "experiment": {
            "p1": 2.0,
            "p2": 1.0,
            "pos2": [2.0, 0.0],
            "velocity2": [0.0, 0.0],
        },
        "predictions": [
            {
                "hypothesis_id": hypothesis_id,
                "target": "/0/y",
                "mean": mean,
                "sigma": 0.01,
            }
            for hypothesis_id, mean in (("H_signed", -10.0), ("H_magnitude", 10.0))
        ],
    }
    parsed = pipeline.ingest_reply(
        _design_block([high_disagreement, sign_design]), round_num=2, max_episode_cost=3
    )
    assert pipeline._design_sign_variations(sign_design) == {"p1", "p2"}
    assert parsed["selected_design"]["design_id"] == "matched_signs"
    scores = {
        row["design_id"]: row for row in parsed["selected_design"]["proposal_scores"]
    }
    assert scores["matched_signs"]["satisfies_sign_audit"] is True
    assert scores["matched_signs"]["sign_audit_gain"] == 2
    pipeline.experiments[0]["local_causal_edges"].extend(
        [
            _anchored_radial_edge(p1=-1.0, p2=1.0, radius=2.0, inward=-0.25),
            _anchored_radial_edge(p1=1.0, p2=-1.0, radius=2.0, inward=-0.25),
        ]
    )
    assert pipeline._missing_sign_audits() == []
    assert not any(
        (
            "matched sign invariance" in blocker
            for blocker in pipeline.finalization_blockers()
        )
    )


class _LinearNoisyExecutor:
    def __init__(self, seed=0):
        self._noise_rng = np.random.default_rng(seed)
        self._vel_noise_rng = np.random.default_rng(seed + 1)

    def run(self, experiments):
        results = []
        for experiment in experiments:
            shared_noise = float(self._noise_rng.normal())
            velocity_noise = float(self._vel_noise_rng.normal())
            results.append(
                {"y": float(experiment["x"]) + shared_noise, "v": velocity_noise}
            )
        return results


class _RadialPowerNoisyExecutor:
    def __init__(self, seed=0, exponent=2.0):
        self._noise_rng = np.random.default_rng(seed)
        self._vel_noise_rng = np.random.default_rng(seed + 1)
        self.exponent = float(exponent)

    def run(self, experiments):
        results = []
        for experiment in experiments:
            position = np.asarray(experiment["pos2"], dtype=float)
            velocity = np.asarray(experiment["velocity2"], dtype=float)
            radius = float(np.linalg.norm(position))
            radial = position / radius
            strength = abs(float(experiment["p1"])) * abs(float(experiment["p2"]))
            acceleration = -strength * radial / radius**self.exponent
            times = list(experiment["measurement_times"])
            position_noise = self._noise_rng.normal(size=2)
            velocity_noise = self._vel_noise_rng.normal(size=2)
            results.append(
                {
                    "measurement_times": times,
                    "pos1": [(0.5 * position_noise).tolist() for _ in times],
                    "pos2": [
                        (
                            position
                            + velocity * float(time)
                            + 0.5 * acceleration * float(time) ** 2
                            + position_noise
                        ).tolist()
                        for time in times
                    ],
                    "velocity1": [velocity_noise.tolist() for _ in times],
                    "velocity2": [
                        (
                            velocity + acceleration * float(time) + velocity_noise
                        ).tolist()
                        for time in times
                    ],
                }
            )
        return results


def test_strict_counterfactual_requires_paired_effect_for_each_candidate():
    pipeline = SCMPipeline("test", strict=True)
    reply = _update_block(_candidate("H1"), _candidate("H2", exponent=2.0))
    reply += _design_block(
        [
            {
                "id": "incomplete_pair",
                "kind": "counterfactual",
                "factual": {"x": 1.0},
                "interventions": [{"id": "do_x_2", "set": {"/x": 2.0}}],
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/paired_effects/0/output_delta/0/y",
                        "mean": 1.0,
                        "sigma": 0.2,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/counterfactuals/0/output/0/y",
                        "mean": 2.0,
                        "sigma": 0.2,
                    },
                ],
            },
            {
                "id": "ordinary_backup",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.2,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/0/y",
                        "mean": 2.0,
                        "sigma": 0.2,
                    },
                ],
            },
        ]
    )
    parsed = pipeline.ingest_reply(reply, round_num=1)
    assert any(("missing H2" in error for error in parsed["protocol_errors"]))


def test_paired_counterfactual_holds_noise_stream_fixed():
    pipeline = SCMPipeline("test")
    reply = _update_block(_candidate("H1"), _candidate("H2", exponent=2.0))
    reply += _design_block(
        [
            {
                "id": "paired",
                "kind": "counterfactual",
                "factual": {"x": 1.0},
                "interventions": [{"id": "do_x_3", "set": {"/x": 3.0}}],
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/counterfactuals/0/output/0/y",
                        "mean": 3.1,
                        "sigma": 2.0,
                    }
                ],
            }
        ]
    )
    pipeline.ingest_reply(reply, round_num=1)
    executor = _LinearNoisyExecutor(seed=7)
    execution = pipeline.execute_pending(executor)
    outcome = execution["outcome"]
    factual_y = outcome["factual"]["output"][0]["y"]
    counterfactual_y = outcome["counterfactuals"][0]["output"][0]["y"]
    factual_v = outcome["factual"]["output"][0]["v"]
    counterfactual_v = outcome["counterfactuals"][0]["output"][0]["v"]
    paired_delta_y = outcome["paired_effects"][0]["output_delta"][0]["y"]
    paired_delta_v = outcome["paired_effects"][0]["output_delta"][0]["v"]
    assert counterfactual_y - factual_y == pytest.approx(2.0)
    assert counterfactual_v == pytest.approx(factual_v)
    assert paired_delta_y == pytest.approx(2.0)
    assert paired_delta_v == pytest.approx(0.0)
    assert outcome["shared_context"]["observation_noise"] == "common_random_numbers"


def test_runtime_reconstructs_noise_cancelled_scaling_and_correspondence():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    reply = _update_block(
        _candidate("H_inverse_square", exponent=2.0),
        _candidate("H_inverse_distance", exponent=1.0),
    )
    reply += _design_block(
        [
            {
                "id": "null_radius_property_audit",
                "kind": "counterfactual",
                "factual": {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [2.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "measurement_times": [0.1],
                },
                "interventions": [
                    {"id": "null", "set": {"/p1": 0.0}},
                    {"id": "near", "set": {"/pos2/0": 1.0}},
                    {"id": "far", "set": {"/pos2/0": 4.0}},
                    {"id": "double_p2", "set": {"/p2": 2.0}},
                ],
                "predictions": [
                    {
                        "hypothesis_id": "H_inverse_square",
                        "target": "/paired_effects/0/output_delta/0/velocity2/0/0",
                        "mean": 0.025,
                        "sigma": 1.0,
                    }
                ],
            }
        ]
    )
    pipeline.ingest_reply(reply, round_num=1)
    execution = pipeline.execute_pending(_RadialPowerNoisyExecutor(seed=17))
    pipeline.validate_pending(execution)
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    scaling = audit["controlled_scaling"]
    assert audit["anchored_response_count"] == 4
    assert audit["graph_residual_rmse"] < 1e-10
    assert scaling["radius_decay_exponent_median"] == pytest.approx(2.0)
    assert scaling["p2_power_exponent_median"] == pytest.approx(1.0)
    assert any(
        ("two-charge Coulomb" in text for text in audit["candidate_correspondences"])
    )
    assert pipeline.metrics()["noise_cancelled_causal_edges"] == 4
    assert pipeline._needs_p2_magnitude_audit() is False
    assert pipeline._needs_start_time_audit() is True


def _anchored_radial_edge(*, p1, p2, radius, inward, start_time=0.0):
    common = {
        "p2": float(p2),
        "pos2": [float(radius), 0.0],
        "velocity2": [0.0, 0.0],
        "start_time": float(start_time),
    }
    return {
        "factual": {"p1": 0.0, **common},
        "counterfactual": {"p1": float(p1), **common},
        "delta_acceleration": [-float(inward), 0.0],
    }


def test_generic_radius_audit_requires_four_anchored_radii_over_eightfold_span():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    edges = [
        _anchored_radial_edge(p1=1.0, p2=1.0, radius=radius, inward=1.0 / radius**2)
        for radius in (1.0, 2.0, 4.0)
    ]
    pipeline.experiments = [{"local_causal_edges": edges}]
    assert pipeline._needs_radius_scale_audit() is True
    assert any(
        ("four-radius" in blocker for blocker in pipeline.finalization_blockers())
    )
    radius_design = {
        "kind": "counterfactual",
        "factual": {"p1": 1.0, "p2": 1.0, "pos2": [1.0, 0.0], "velocity2": [0.0, 0.0]},
        "interventions": [{"id": "far", "set": {"/pos2/0": 4.0}}],
    }
    assert pipeline._design_has_radius_variation(radius_design) is True
    edges.append(_anchored_radial_edge(p1=1.0, p2=1.0, radius=0.5, inward=4.0))
    assert pipeline._needs_radius_scale_audit() is False
    assert not any(
        ("four-radius" in blocker for blocker in pipeline.finalization_blockers())
    )


def test_first_radius_scan_must_straddle_factual_scale():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    one_sided = {
        "kind": "counterfactual",
        "factual": {"p1": 1.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]},
        "interventions": [
            {"id": "r4", "set": {"/pos2/0": 4.0}},
            {"id": "r8", "set": {"/pos2/0": 8.0}},
        ],
    }
    bidirectional = {
        **one_sided,
        "interventions": [
            {"id": "r1", "set": {"/pos2/0": 1.0}},
            {"id": "r8", "set": {"/pos2/0": 8.0}},
        ],
    }
    assert pipeline._design_has_radius_variation(one_sided) is True
    assert pipeline._design_advances_radius_scale_audit(one_sided) is False
    assert pipeline._design_advances_radius_scale_audit(bidirectional) is True


def test_causal_selector_prefers_new_near_scale_over_redundant_clock_axis():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    pipeline.ingest_reply(
        _update_block(_candidate("H1"), _candidate("H2", exponent=2.0)), round_num=1
    )
    pipeline.experiments = [
        {
            "local_causal_edges": [
                _anchored_radial_edge(
                    p1=p1, p2=p2, radius=radius, inward=p1 / p2 / radius
                )
                for p1, p2, radius in (
                    (1.0, 1.0, 2.0),
                    (1.0, 1.0, 4.0),
                    (1.0, 1.0, 8.0),
                    (-1.0, 1.0, 2.0),
                    (1.0, -1.0, 2.0),
                    (1.0, 2.0, 2.0),
                )
            ]
        }
    ]
    common_predictions = [
        {
            "hypothesis_id": "H1",
            "target": "/paired_effects/0/output_delta/0/velocity2/-1/0",
            "mean": 0.0,
            "sigma": 0.1,
        },
        {
            "hypothesis_id": "H2",
            "target": "/paired_effects/0/output_delta/0/velocity2/-1/0",
            "mean": 1.0,
            "sigma": 0.1,
        },
    ]
    clock = {
        "id": "clock_only",
        "kind": "counterfactual",
        "factual": {
            "p1": 1.0,
            "p2": 1.0,
            "pos2": [2.0, 0.0],
            "velocity2": [0.0, 0.0],
            "start_time": 0.0,
        },
        "interventions": [
            {"id": "phase1", "set": {"/start_time": 1.0}},
            {"id": "phase2", "set": {"/start_time": 2.0}},
        ],
        "predictions": common_predictions,
    }
    near = {
        "id": "new_near_scale",
        "kind": "counterfactual",
        "factual": {"p1": 1.0, "p2": 1.0, "pos2": [2.0, 0.0], "velocity2": [0.0, 0.0]},
        "interventions": [{"id": "radius1", "set": {"/pos2/0": 1.0}}],
        "predictions": common_predictions,
    }
    farther = {
        **near,
        "id": "same_asymptote_farther",
        "interventions": [{"id": "radius16", "set": {"/pos2/0": 16.0}}],
    }
    selected = pipeline._select_design([clock, near])
    scores = {row["design_id"]: row for row in selected.proposal_scores}
    assert pipeline._needs_start_time_audit() is True
    assert pipeline._needs_radius_scale_audit() is True
    assert scores["clock_only"]["satisfies_start_time_audit"] is True
    assert scores["clock_only"]["satisfies_radius_scale_audit"] is False
    assert scores["new_near_scale"]["satisfies_radius_scale_audit"] is True
    assert pipeline._design_advances_radius_scale_audit(farther) is False
    assert selected.design_id == "new_near_scale"


def test_multiplicative_property_exponent_overrides_false_scale_crossover():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    radius_response = {0.25: 16.0, 0.5: 4.0, 1.0: 1.0, 2.0: 0.5, 4.0: 0.25}
    edges = [
        _anchored_radial_edge(p1=1.0, p2=1.0, radius=radius, inward=inward)
        for radius, inward in radius_response.items()
    ]
    edges.extend(
        [
            _anchored_radial_edge(p1=1.0, p2=2.0, radius=4.0, inward=0.5),
            _anchored_radial_edge(p1=-1.0, p2=1.0, radius=4.0, inward=0.25),
            _anchored_radial_edge(p1=1.0, p2=-1.0, radius=4.0, inward=0.25),
        ]
    )
    pipeline.experiments = [{"local_causal_edges": edges}]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert audit["controlled_scaling"]["p2_power_exponent_median"] == pytest.approx(1.0)
    assert "Coulomb-like central force" in correspondence
    assert "fixed opposite polarity" in correspondence
    assert "always attractive" in correspondence
    assert "compactified" not in correspondence
    assert "screened/Helmholtz" not in correspondence


def test_radius_crossover_pools_repeated_clock_intervals():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    edges = []
    for start_time in (0.0, 1.0, 2.0):
        for radius in (0.25, 0.5, 1.0):
            inward = 1.0 / radius**2
            edges.append(
                _anchored_radial_edge(
                    p1=1.0, p2=1.0, radius=radius, inward=inward, start_time=start_time
                )
            )
    for radius in (2.0, 4.0, 8.0):
        inward = 1.0 / radius
        edges.append(
            _anchored_radial_edge(p1=1.0, p2=1.0, radius=radius, inward=inward)
        )
    pipeline.experiments = [{"local_causal_edges": edges}]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    scaling = audit["controlled_scaling"]
    intervals = scaling["radius_local_decay"]
    assert len(intervals) == 5
    assert [item["replicate_count"] for item in intervals[:2]] == [3, 3]
    assert scaling["near_range_decay_q"] > 1.5
    assert scaling["far_range_decay_q"] < 1.5
    assert any(
        (
            "extra spatial dimension compactified" in text
            for text in audit["candidate_correspondences"]
        )
    )
    assert not any(
        ("fractional-Laplacian" in text for text in audit["candidate_correspondences"])
    )


def test_fractional_correspondence_preserves_nonlocal_long_range_interpretation():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    edges = [
        _anchored_radial_edge(p1=1.0, p2=1.0, radius=1.0, inward=1.0),
        _anchored_radial_edge(p1=1.0, p2=1.0, radius=2.0, inward=0.25),
        _anchored_radial_edge(p1=1.0, p2=2.0, radius=2.0, inward=0.125),
    ]
    pipeline.experiments = [{"local_causal_edges": edges}]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert "fractional-Laplacian" in correspondence
    assert "enhanced long-range coupling" in correspondence
    assert "not an ordinary local central-force mechanism" in correspondence
    pipeline.final_claim = None
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline
    audited = agent._audited_explanation(
        "A contradictory ordinary inverse-square mechanism."
    )
    assert audited.startswith("Structured SCM conclusion.")
    assert "alpha≈0.5" in audited
    assert "non-local spatial operator" in audited
    assert "contradictory ordinary" not in audited


def test_compact_image_sum_fit_recovers_scale_from_causal_responses():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    compact_radius = 0.5
    field_scale = 1.0
    circumference = 2.0 * np.pi * compact_radius
    images = np.arange(-128, 129, dtype=float)
    edges = []
    for radius in (0.3125, 0.625, 1.25, 2.5, 5.0, 10.0):
        unit_kernel = (
            circumference
            / (4.0 * np.pi)
            * np.sum(radius / (radius**2 + (images * circumference) ** 2) ** 1.5)
        )
        edges.append(
            _anchored_radial_edge(
                p1=1.0, p2=1.0, radius=radius, inward=field_scale * unit_kernel
            )
        )
        if radius == 2.5:
            edges.append(
                _anchored_radial_edge(
                    p1=1.0,
                    p2=2.0,
                    radius=radius,
                    inward=field_scale * unit_kernel / 2.0,
                )
            )
    pipeline.experiments = [{"local_causal_edges": edges}]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    fit = audit["mechanism_fits"]["compactified_image_sum"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert fit["compactification_radius_R"] == pytest.approx(compact_radius, rel=0.02)
    assert fit["field_scale_G"] == pytest.approx(field_scale, rel=0.02)
    assert fit["relative_log_rmse"] < 0.001
    assert "quantitatively selects" in correspondence
    assert "slope-transition midpoint" in correspondence
    compiled = pipeline.deductive_compactified_law()
    assert compiled is not None
    executable_check = pipeline.executable_causal_check(compiled["source"])
    assert executable_check["passed"] is True
    assert executable_check["max_radius_median_relative_vector_error"] < 0.03
    pipeline.final_claim = {
        "source_response_roles": "p1 is source strength and p2 is the inertial response denominator"
    }
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline
    audited = agent._audited_explanation(
        "A contradictory generic noninteger power law."
    )
    assert audited.startswith("Structured SCM conclusion.")
    assert "1/r^2 at short range and 1/r at long range" in audited
    assert "contradictory generic" not in audited


def test_ordered_change_point_keeps_one_decisive_near_interval():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    responses = {
        1.0: 0.2667119675814142,
        2.0: 0.08766050684918028,
        4.0: 0.039808710618813734,
        8.0: 0.019742435941495464,
        16.0: 0.009653843217345946,
    }
    pipeline.experiments = [
        {
            "local_causal_edges": [
                _anchored_radial_edge(p1=1.0, p2=1.0, radius=radius, inward=inward)
                for radius, inward in responses.items()
            ]
            + [
                _anchored_radial_edge(
                    p1=1.0, p2=2.0, radius=4.0, inward=responses[4.0] / 2.0
                )
            ]
        }
    ]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    change = audit["mechanism_fits"]["ordered_radius_change_point"]
    fit = audit["mechanism_fits"]["compactified_image_sum"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert change["near_decay_q"] == pytest.approx(1.60528, rel=0.001)
    assert change["far_decay_q"] == pytest.approx(1.03212, rel=0.02)
    assert change["transition_radius"] == pytest.approx(2.0)
    assert fit["compactification_radius_R"] == pytest.approx(0.508, rel=0.03)
    assert fit["relative_log_rmse"] < 0.02
    assert "quantitatively selects" in correspondence
    assert "extra spatial dimension compactified" in correspondence
    assert "ordinary 2D Poisson" not in correspondence


def test_weak_crossover_explanation_keeps_correspondence_not_generic_claim():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    pipeline.final_claim = {
        "source_response_roles": "p1 is source and p2 is inertial response",
        "scalar_magnitude_law": "a contradictory softened inverse-square approximation",
        "time_and_scale_regimes": "only one local regime",
    }
    pipeline.fit_attempts = [
        {"error": None, "fitted_params": {"G": 0.25, "L": 3.14, "eps": 0.04}}
    ]
    pipeline.state_summary = lambda: {
        "noise_cancelled_causal_audit": {
            "candidate_correspondences": [
                "one extra spatial dimension compactified on a circle; the image sum has a 1/r^2 short-range and 1/r long-range dimensional crossover, with L=2*pi*R"
            ],
            "controlled_scaling": {},
        }
    }
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline
    audited = agent._audited_explanation("generic approximation")
    assert "1/r^2 short-range and 1/r long-range" in audited
    assert "R=L/(2*pi)≈" in audited
    assert "softened inverse-square approximation" not in audited
    assert "generic approximation" not in audited


def test_unsupported_weak_correspondence_cannot_override_final_claim():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    pipeline.final_claim = {
        "operator_correspondence_check": "The supported operator is a Coulomb-like inverse-square field.",
        "source_response_roles": "p1 and p2 are multiplicative magnitude controls.",
        "time_and_scale_regimes": "static over the tested phases",
    }
    pipeline.state_summary = lambda: {
        "noise_cancelled_causal_audit": {
            "candidate_correspondences": [
                "the force changes slope; compare a single extra spatial dimension compactified on a circle"
            ],
            "controlled_scaling": {},
        },
        "metrics": {"selected_candidate_evidence_supported": False},
    }
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline
    audited = agent._audited_explanation(
        "The executable conclusion is a static inverse-square central law."
    )
    assert "Coulomb-like inverse-square" in audited
    assert "static inverse-square central law" in audited
    assert "extra spatial dimension" not in audited


def test_executable_causal_check_rejects_local_response_as_global_coefficient():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    edges = [
        _anchored_radial_edge(p1=1.0, p2=1.0, radius=radius, inward=1.0 / radius**2)
        for radius in (1.0, 2.0, 4.0, 8.0)
    ]
    edges.extend(
        [
            _anchored_radial_edge(p1=1.0, p2=2.0, radius=4.0, inward=0.125),
            _anchored_radial_edge(p1=-1.0, p2=1.0, radius=4.0, inward=0.0625),
            _anchored_radial_edge(p1=1.0, p2=-1.0, radius=4.0, inward=0.0625),
        ]
    )
    pipeline.experiments = [{"local_causal_edges": edges}]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert audit["controlled_scaling"][
        "radial_power_coefficient_median"
    ] == pytest.approx(1.0)
    assert "global coupling C≈1" in correspondence
    law_template = '\ndef discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):\n    coefficient = params.get("C", COEFFICIENT)\n    dx = pos1[0] - pos2[0]\n    dy = pos1[1] - pos2[1]\n    radius = (dx * dx + dy * dy) ** 0.5\n    scale = coefficient * abs(p1) * abs(p2) / radius ** 3\n    acceleration = [scale * dx, scale * dy]\n    return (\n        list(pos2),\n        [\n            velocity2[0] + duration * acceleration[0],\n            velocity2[1] + duration * acceleration[1],\n        ],\n    )\n'
    good = pipeline.executable_causal_check(law_template.replace("COEFFICIENT", "1.0"))
    bad = pipeline.executable_causal_check(
        law_template.replace("COEFFICIENT", "0.0625")
    )
    assert good["available"] is True
    assert good["passed"] is True
    assert good["median_relative_vector_error"] < 1e-09
    assert bad["available"] is True
    assert bad["passed"] is False
    assert bad["median_relative_vector_error"] == pytest.approx(0.9375)


def test_velocity_counterfactual_removes_initial_kinematic_delta():
    execution = {
        "kind": "counterfactual",
        "outcome": {
            "factual": {
                "input": {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [3.0, 0.0],
                    "velocity2": [1.0, 0.0],
                    "measurement_times": [0.1],
                },
                "output": [{"measurement_times": [0.1], "velocity2": [[0.9, 0.0]]}],
            },
            "counterfactuals": [
                {
                    "id": "reverse_velocity",
                    "input": {
                        "p1": 1.0,
                        "p2": 1.0,
                        "pos2": [3.0, 0.0],
                        "velocity2": [-1.0, 0.0],
                        "measurement_times": [0.1],
                    },
                }
            ],
            "paired_effects": [
                {
                    "id": "reverse_velocity",
                    "output_delta": [{"velocity2": [[-2.0, 0.0]]}],
                }
            ],
        },
    }
    edges = SCMPipeline._local_counterfactual_edges(execution)
    assert len(edges) == 1
    assert edges[0]["initial_velocity_delta"] == [-2.0, 0.0]
    assert edges[0]["delta_acceleration"] == pytest.approx([0.0, 0.0])


def test_single_cadence_legacy_batch_recovers_three_phase_time_response():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    omega = np.pi / 2.0
    inputs = [
        {
            "p1": 0.0,
            "p2": 1.0,
            "pos2": [4.0, 0.0],
            "velocity2": [0.0, 0.0],
            "measurement_times": [0.2],
            "start_time": 0.0,
        }
    ]
    outputs = [{"measurement_times": [0.2], "velocity2": [[0.0, 0.0]]}]
    for start_time in (0.0, 1.0, 2.0):
        inward = 0.2 * np.cos(omega * start_time)
        inputs.append(
            {
                "p1": 1.0,
                "p2": 1.0,
                "pos2": [4.0, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": [0.2],
                "start_time": start_time,
            }
        )
        outputs.append(
            {"measurement_times": [0.2], "velocity2": [[-0.2 * inward, 0.0]]}
        )
    pipeline.experiments = [
        {"kind": "intervention", "input": inputs, "output": outputs}
    ]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    fit = audit["mechanism_fits"]["absolute_time_sinusoid"]
    assert audit["paired_edge_count"] == 0
    assert audit["direct_single_cadence_edge_count"] == 3
    assert fit["fit_mode"] == "three_phase_antisymmetric"
    assert fit["angular_frequency_omega"] == pytest.approx(omega)
    assert fit["period"] == pytest.approx(4.0)
    assert fit["relative_rmse"] < 1e-10
    assert [
        item["angular_frequency_omega"] for item in fit["frequency_aliases"][:3]
    ] == pytest.approx([omega, 3.0 * omega, 5.0 * omega])
    assert pipeline._needs_start_time_audit() is True


def test_single_cadence_batch_reuses_prior_paired_null_anchor():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    pipeline.experiments = [
        {
            "local_causal_edges": [
                _anchored_radial_edge(p1=1.0, p2=1.0, radius=4.0, inward=0.25)
            ]
        },
        {
            "kind": "intervention",
            "input": [
                {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [1.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "measurement_times": [0.2],
                },
                {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [16.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "measurement_times": [0.2],
                },
            ],
            "output": [
                {"measurement_times": [0.2], "velocity2": [[-0.2, 0.0]]},
                {"measurement_times": [0.2], "velocity2": [[-0.0125, 0.0]]},
            ],
        },
    ]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    radii = {row["radius"] for row in audit["responses"] if abs(row["p1"]) > 1e-12}
    assert audit["paired_edge_count"] == 1
    assert audit["direct_single_cadence_edge_count"] == 2
    assert radii == {1.0, 4.0, 16.0}


def test_identified_time_scm_compiles_to_evidence_preserving_law():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    omega = np.pi / 2.0
    inputs = [
        {
            "p1": 0.0,
            "p2": 1.0,
            "pos2": [4.0, 0.0],
            "velocity2": [0.0, 0.0],
            "measurement_times": [0.2],
            "start_time": 0.0,
        }
    ]
    outputs = [{"measurement_times": [0.2], "velocity2": [[0.0, 0.0]]}]
    cases = [
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 2.0, 0.0),
        (1.0, 1.0, 4.0, 0.0),
        (1.0, 1.0, 8.0, 0.0),
        (-1.0, 1.0, 4.0, 0.0),
        (1.0, -1.0, 4.0, 0.0),
        (1.0, 2.0, 4.0, 0.0),
        (1.0, 1.0, 4.0, 1.0),
        (1.0, 1.0, 4.0, 2.0),
    ]
    for p1, p2, radius, start_time in cases:
        coupling = 0.8 * np.cos(omega * start_time)
        acceleration_x = -coupling * (p1 / p2) / radius
        inputs.append(
            {
                "p1": p1,
                "p2": p2,
                "pos2": [radius, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": [0.2],
                "start_time": start_time,
            }
        )
        outputs.append(
            {"measurement_times": [0.2], "velocity2": [[0.2 * acceleration_x, 0.0]]}
        )
    pipeline.experiments = [
        {"kind": "intervention", "input": inputs, "output": outputs}
    ]
    compiled = pipeline.deductive_time_modulated_law()
    assert compiled is not None
    assert "signed p1/p2" in compiled["deduction"]
    assert '"q": {"init"' not in compiled["source"]
    assert '"omega": {"init"' not in compiled["source"]
    assert '"phase": {"init"' not in compiled["source"]
    assert len(pipeline.deductive_time_modulated_laws()) == 5
    check = pipeline.executable_causal_check(compiled["source"])
    assert check["available"] is True
    assert check["passed"] is True
    assert check["strongly_identified_mechanism"] is False
    assert check["median_relative_vector_error"] < 0.05
    assert check["p80_relative_vector_error"] < 0.1


def test_sparse_two_interval_dimensional_crossover_is_not_fractional():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    responses = {
        1.0: 0.5335572132872897,
        2.0: 0.17533820234683173,
        4.0: 0.07961877155794743,
    }
    pipeline.experiments = [
        {
            "local_causal_edges": [
                _anchored_radial_edge(p1=2.0, p2=1.0, radius=radius, inward=inward)
                for radius, inward in responses.items()
            ]
        }
    ]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert audit["controlled_scaling"]["near_range_decay_q"] == pytest.approx(
        1.6055026174536453
    )
    assert audit["controlled_scaling"]["far_range_decay_q"] == pytest.approx(
        1.1389598446387672
    )
    assert "extra spatial dimension" in correspondence
    assert "fractional-Laplacian" not in correspondence


def test_high_training_replay_loss_blocks_unchecked_final_code():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline

    def fake_fit(round_num, law_source, scratch):
        scratch["mse_fit_output"] = {
            "error": None,
            "loss_before": 0.02,
            "loss_after": 0.02,
            "fitted_params": {},
            "declared_params": {},
            "n_training": 3,
            "training_mode": "conversation",
        }

    agent._run_mse_fit = fake_fit
    result = agent._evaluate_final_law_consistency(
        3, "def discovered_law(*args, **kwargs): return args"
    )
    assert result["training_check"]["available"] is True
    assert result["training_check"]["passed"] is False
    assert result["training_check"]["loss_relative_to_noise_variance"] == pytest.approx(
        2.0
    )
    assert result["causal_check"]["available"] is True
    assert result["causal_check"]["passed"] is False
    assert "replay loss" in result["causal_check"]["reason"]


def test_final_code_check_requests_repair_without_new_simulator_episode():
    pipeline = SCMPipeline("opaque_internal_id")
    pipeline.state_summary = lambda: {
        "noise_cancelled_causal_audit": {
            "candidate_correspondences": ["global coupling C≈1"],
            "controlled_scaling": {"radial_power_coefficient_median": 1.0},
            "mechanism_fits": {},
        }
    }
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline
    agent.model = "test-model"
    agent.max_tokens = 1024
    agent._system = "test system"
    agent._law_stub = "def discovered_law(*args, **kwargs): ..."
    initial = {
        "fit": {"fitted_params": {"C": 0.2}},
        "causal_check": {
            "available": True,
            "passed": False,
            "median_relative_vector_error": 0.8,
            "p80_relative_vector_error": 0.8,
        },
    }
    repaired = {
        "fit": {"fitted_params": {"C": 1.0}},
        "causal_check": {
            "available": True,
            "passed": True,
            "median_relative_vector_error": 0.01,
            "p80_relative_vector_error": 0.02,
        },
    }
    checks = iter((initial, repaired))
    agent._evaluate_final_law_consistency = lambda round_num, law_source: next(checks)
    messages = []
    round_entry = {}
    with patch(
        "scienceagent.agent.llm_client.complete",
        return_value="<final_law>GOOD_LAW</final_law><explanation>repaired normalization</explanation>",
    ) as complete:
        law, explanation = agent._repair_final_law_if_needed(
            round_num=4,
            law_source="BAD_LAW",
            raw_explanation="old explanation",
            messages=messages,
            round_entry=round_entry,
        )
    assert law == "GOOD_LAW"
    assert explanation == "repaired normalization"
    assert complete.call_count == 1
    assert round_entry["final_code_repaired"] is True
    assert len(round_entry["final_code_checks"]) == 2
    assert "global coupling C≈1" in messages[-2]["content"]


def test_final_code_check_selects_time_alias_by_existing_trajectory_loss():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    pipeline.deductive_time_modulated_laws = lambda: [
        {
            "source": "alias_zero",
            "deduction": "lowest-frequency alias",
            "time_alias": {
                "alias_index": 0,
                "angular_frequency_omega": 0.5,
                "phase": 0.0,
                "period": 4.0 * np.pi,
            },
        },
        {
            "source": "alias_one",
            "deduction": "trajectory-selected alias",
            "time_alias": {
                "alias_index": 1,
                "angular_frequency_omega": 1.5,
                "phase": 0.0,
                "period": 4.0 * np.pi / 3.0,
            },
        },
    ]
    agent = object.__new__(DiscoveryAgent)
    agent.scm_pipeline = pipeline
    losses = {"initial": 1.0, "alias_zero": 0.8, "alias_one": 0.4}

    def fake_check(round_num, source):
        return {
            "fit": {"loss_after": losses[source], "error": None},
            "causal_check": {
                "available": True,
                "passed": True,
                "median_relative_vector_error": 0.01,
                "p80_relative_vector_error": 0.02,
            },
            "training_check": {
                "available": True,
                "passed": True,
                "loss_relative_to_noise_variance": losses[source],
            },
        }

    agent._evaluate_final_law_consistency = fake_check
    round_entry = {}
    repaired, explanation = agent._repair_final_law_if_needed(
        round_num=5,
        law_source="initial",
        raw_explanation="time law",
        messages=[],
        round_entry=round_entry,
    )
    assert repaired == "alias_one"
    assert explanation == "time law"
    assert round_entry["final_code_repaired"] is True
    assert pipeline.time_alias_selection["selected_alias_index"] == 1
    assert pipeline.time_alias_selection["simulator_episodes_added"] == 0
    assert len(pipeline.time_alias_selection["candidate_aliases"]) == 2


def test_screened_helmholtz_fit_recovers_length_from_causal_responses():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    screening_length = 2.0
    field_scale = 1.0
    edges = []
    for radius in (1.0, 2.0, 4.0, 8.0, 16.0):
        inward = (
            field_scale
            * bessel_k1(radius / screening_length)
            / (2.0 * np.pi * screening_length)
        )
        edges.append(
            _anchored_radial_edge(p1=1.0, p2=1.0, radius=radius, inward=inward)
        )
    pipeline.experiments = [{"local_causal_edges": edges}]
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    fit = audit["mechanism_fits"]["screened_helmholtz"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert fit["screening_length_lambda"] == pytest.approx(screening_length, rel=0.02)
    assert fit["field_scale_G"] == pytest.approx(field_scale, rel=0.02)
    assert fit["relative_log_rmse"] < 0.001
    assert "quantitatively selects" in correspondence
    assert "screening length lambda" in correspondence
    assert "is not the screening length" in correspondence


def test_absolute_time_sinusoid_fit_recovers_frequency_and_completes_clock_audit():
    pipeline = SCMPipeline("opaque_internal_id", observation_noise_std=0.1)
    omega = np.pi / 2.0
    edges = [
        _anchored_radial_edge(
            p1=1.0,
            p2=1.0,
            radius=4.0,
            inward=0.2 * np.cos(omega * start_time),
            start_time=start_time,
        )
        for start_time in (0.0, 1.0, 2.0)
    ]
    pipeline.experiments = [{"local_causal_edges": edges}]
    three_phase_fit = pipeline.state_summary()["noise_cancelled_causal_audit"][
        "mechanism_fits"
    ]["absolute_time_sinusoid"]
    assert three_phase_fit["fit_mode"] == "three_phase_antisymmetric"
    assert len(three_phase_fit["frequency_aliases"]) == 5
    assert pipeline._needs_start_time_audit() is True
    edges.append(
        _anchored_radial_edge(
            p1=1.0, p2=1.0, radius=4.0, inward=0.2 * np.cos(omega * 3.0), start_time=3.0
        )
    )
    audit = pipeline.state_summary()["noise_cancelled_causal_audit"]
    fit = audit["mechanism_fits"]["absolute_time_sinusoid"]
    correspondence = " ".join(audit["candidate_correspondences"])
    assert fit["angular_frequency_omega"] == pytest.approx(omega, rel=0.02)
    assert fit["period"] == pytest.approx(4.0, rel=0.02)
    assert fit["amplitude"] == pytest.approx(0.2, rel=0.02)
    assert fit["relative_rmse"] < 0.001
    assert pipeline._needs_start_time_audit() is False
    assert "sinusoidal coupling is quantitatively identified" in correspondence
    assert "start_time+t" in correspondence


def test_prediction_validation_updates_posterior_and_records_counterexample():
    pipeline = SCMPipeline("test", observation_noise_std=0.1)
    reply = _update_block(_candidate("H_good"), _candidate("H_bad", exponent=2.0))
    reply += _design_block(
        [
            {
                "id": "decisive",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H_good",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.2,
                    },
                    {
                        "hypothesis_id": "H_bad",
                        "target": "/0/y",
                        "mean": 5.0,
                        "sigma": 0.2,
                    },
                ],
            }
        ]
    )
    pipeline.ingest_reply(reply, round_num=1)
    execution = {
        "kind": "intervention",
        "outcome": [{"y": 1.05}],
        "experiment_input": [{"x": 1.0}],
        "experiment_output": [{"y": 1.05}],
    }
    validation = pipeline.validate_pending(execution)
    status = {
        record["hypothesis_id"]: record["status"] for record in validation["records"]
    }
    assert status == {"H_good": "success", "H_bad": "counterexample"}
    assert validation["posterior"]["H_good"] > 0.999
    assert pipeline.metrics()["counterexamples"] == 1
    assert all(
        (record["causal_evidence_weight"] == 0.25 for record in validation["records"])
    )


def test_retirement_requires_replicated_paired_or_repeated_design_evidence():
    pipeline = SCMPipeline("test")
    pipeline.ingest_reply(
        _update_block(_candidate("H1"), _candidate("H2", exponent=2.0)), round_num=1
    )
    retirement = (
        "<scm_update>"
        + json.dumps(
            {
                "mode": "revise",
                "patches": [
                    {
                        "candidate_id": "H1",
                        "operation": "retire_candidate",
                        "payload": {},
                        "reason": "one uncertain batch",
                    }
                ],
            }
        )
        + "</scm_update>"
    )
    parsed = pipeline.ingest_reply(retirement, round_num=2)
    assert pipeline.candidates["H1"]["status"] == "active"
    assert "retire_candidate requires" in parsed["protocol_errors"][0]
    assert pipeline.patches[-1]["applied"] is False


def test_one_unpaired_counterexample_cannot_retire_structural_candidate():
    pipeline = SCMPipeline("test")
    pipeline.ingest_reply(
        _update_block(_candidate("H1"), _candidate("H2", exponent=2.0)), round_num=1
    )
    pipeline.evidence = [
        {
            "hypothesis_id": "H1",
            "design_id": "noisy_absolute",
            "target": "/0/velocity2/-1/0",
            "status": "counterexample",
            "relative_log_evidence": -4.0,
        }
    ]
    assert pipeline._retirement_is_supported("H1") is False
    pipeline.evidence.append(
        {
            "hypothesis_id": "H1",
            "design_id": "replicated_pair",
            "target": "/paired_effects/0/output_delta/0/velocity2/-1/0",
            "status": "counterexample",
            "relative_log_evidence": 0.0,
        }
    )
    assert pipeline._retirement_is_supported("H1") is False
    pipeline.evidence[-1]["relative_log_evidence"] = -2.0
    assert pipeline._retirement_is_supported("H1") is True


def test_posterior_uses_relative_evidence_and_penalizes_needless_complexity():
    simple = _candidate("H_simple")
    complex_model = _candidate("H_complex")
    complex_model["parameters"]["gamma"] = {"estimate": 0.0, "lower": 0.0, "upper": 1.0}
    complex_model["edges"].append({"source": "v", "target": "a"})
    complex_model["mechanisms"][0][
        "equation"
    ] = "a=-k/r-gamma*v  # simple model plus optional drag"
    pipeline = SCMPipeline("test", complexity_penalty=0.5)
    reply = _update_block(simple, complex_model)
    reply += _design_block(
        [
            {
                "id": "tie",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H_simple",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.2,
                    },
                    {
                        "hypothesis_id": "H_complex",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 0.2,
                    },
                    {
                        "hypothesis_id": "H_complex",
                        "target": "/0/z",
                        "mean": 2.0,
                        "sigma": 0.1,
                    },
                ],
            }
        ]
    )
    pipeline.ingest_reply(reply, round_num=1)
    prior = pipeline.posterior()
    execution = {
        "kind": "intervention",
        "outcome": [{"y": 1.0, "z": 2.0}],
        "experiment_input": [{"x": 1.0}],
        "experiment_output": [{"y": 1.0, "z": 2.0}],
    }
    validation = pipeline.validate_pending(execution)
    posterior = validation["posterior"]
    assert prior["H_simple"] > prior["H_complex"]
    assert posterior["H_simple"] == pytest.approx(prior["H_simple"])
    assert posterior["H_complex"] == pytest.approx(prior["H_complex"])
    singleton = next((row for row in validation["records"] if row["target"] == "/0/z"))
    assert singleton["relative_log_evidence"] == 0.0


def test_near_identical_predictions_share_structural_evidence():
    pipeline = SCMPipeline("test", complexity_penalty=0.0)
    reply = _update_block(_candidate("H1"), _candidate("H2", exponent=2.0))
    reply += _design_block(
        [
            {
                "id": "not_discriminating",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H1",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 1.0,
                    },
                    {
                        "hypothesis_id": "H2",
                        "target": "/0/y",
                        "mean": 1.1,
                        "sigma": 1.0,
                    },
                ],
            }
        ]
    )
    pipeline.ingest_reply(reply, round_num=1)
    before = pipeline.posterior()
    validation = pipeline.validate_pending(
        {
            "kind": "intervention",
            "outcome": [{"y": 1.1}],
            "experiment_input": [{"x": 1.0}],
            "experiment_output": [{"y": 1.1}],
        }
    )
    assert validation["records"][0]["relative_log_evidence"] == 0.0
    assert validation["records"][1]["relative_log_evidence"] == 0.0
    assert validation["posterior"] == pytest.approx(before)


def test_complete_state_is_strict_json_serializable(tmp_path):
    pipeline = SCMPipeline("test")
    pipeline.ingest_reply(
        _update_block(_candidate("H1"), _candidate("H2", exponent=2.0)), round_num=1
    )
    pipeline.finalize(
        "def discovered_law(*args):\n    return args\n", "A test mechanism."
    )
    encoded = json.dumps(pipeline.to_dict(), allow_nan=False)
    assert '"candidates"' in encoded
    output = tmp_path / "state.json"
    pipeline.write_json(output)
    assert json.loads(output.read_text())["final_law"].startswith("def discovered_law")


def test_scm_and_counterfactual_rounds_feed_evaluator_parameter_fit():
    log = [
        {
            "action": "scm_experiment",
            "experiment_input": [{"x": 1.0}],
            "experiment_output": [{"measurement_times": [1.0], "y": [2.0]}],
        },
        {
            "action": "counterfactual",
            "experiment_output": {
                "factual": {
                    "input": {"x": 2.0},
                    "output": [{"measurement_times": [1.0], "y": [4.0]}],
                },
                "counterfactuals": [
                    {
                        "input": {"x": 3.0},
                        "output": [{"measurement_times": [1.0], "y": [6.0]}],
                    }
                ],
            },
        },
    ]
    training = _extract_training_trajectories(log)
    assert [row["input"]["x"] for row in training] == [1.0, 2.0, 3.0]
    assert "_paired_factual" not in training[1]
    assert training[2]["_paired_factual"]["input"]["x"] == 2.0


def test_discovery_agent_rejects_early_final_with_mandatory_audit_blocker():
    pipeline = SCMPipeline("synthetic", strict=True)
    executor = _LinearNoisyExecutor(seed=13)
    first = _update_block(_candidate("H_linear"), _candidate("H_constant"))
    first += _design_block(
        [
            {
                "id": "r1",
                "kind": "intervention",
                "experiment": {"x": 1.0},
                "predictions": [
                    {
                        "hypothesis_id": "H_linear",
                        "target": "/0/y",
                        "mean": 1.0,
                        "sigma": 2.0,
                    },
                    {
                        "hypothesis_id": "H_constant",
                        "target": "/0/y",
                        "mean": 0.0,
                        "sigma": 2.0,
                    },
                ],
            }
        ]
    )
    final = '\n<scm_final>\n{"selected_candidate_id":"H_linear",\n "remaining_alternatives":["H_constant"],\n "identified_mechanism":"y follows do(x)",\n "operator_or_symmetry":"translation-invariant scalar response",\n "source_response_roles":"x is the intervention and y is the response",\n "scalar_magnitude_law":"the response change equals the input change",\n "vector_law":"y_next = x",\n "time_and_scale_regimes":"static",\n "remaining_uncertainty":"none",\n "evidence_summary":"the intervention separated candidates",\n "noise_cancelled_scaling_check":"linear paired effect",\n "operator_correspondence_check":"scalar linear response",\n "evidence_consistency_check":"code preserves linearity",\n "numerical_robustness_check":"no singularity"}\n</scm_final>\n<final_law>\ndef discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n    return pos2, velocity2\n</final_law>\n<explanation>A linear response.</explanation>\n'
    replies = iter([first, final, final])
    agent = DiscoveryAgent(
        model="fake",
        executor=executor,
        verbose=False,
        max_rounds=3,
        min_rounds=2,
        scm_pipeline=pipeline,
    )
    with patch(
        "scienceagent.llm_client.complete", side_effect=lambda **kwargs: next(replies)
    ), patch.object(
        pipeline,
        "finalization_blockers",
        side_effect=lambda: ["four-radius noise-cancelled scale scan"]
        if pipeline.round == 2
        else [],
    ):
        law = agent.run()
    assert law is not None
    assert [entry["action"] for entry in agent.conversation_log] == [
        "scm_experiment",
        "warning",
        "final_law",
    ]
    assert agent.conversation_log[1]["finalization_blockers"] == [
        "four-radius noise-cancelled scale scan"
    ]
    assert "rejected_final_law" in agent.conversation_log[1]
    assert pipeline.final_claim["selected_candidate_id"] == "H_linear"


def test_strict_final_claim_requires_scientific_consistency_fields():
    pipeline = SCMPipeline(world="toy", strict=True)
    pipeline.ingest_reply(
        _update_block(_candidate("H_linear"), _candidate("H_constant")), round_num=1
    )
    parsed = pipeline.ingest_reply(
        '\n<scm_final>\n{"selected_candidate_id":"H_linear","identified_mechanism":"linear response"}\n</scm_final>\n<final_law>def discovered_law(): return 1</final_law>\n',
        round_num=2,
    )
    assert any(("operator_or_symmetry" in error for error in parsed["protocol_errors"]))
    assert any(
        ("evidence_consistency_check" in error for error in parsed["protocol_errors"])
    )
    assert any(
        ("numerical_robustness_check" in error for error in parsed["protocol_errors"])
    )
    assert any(
        (
            "noise_cancelled_scaling_check" in error
            for error in parsed["protocol_errors"]
        )
    )
    assert any(
        (
            "operator_correspondence_check" in error
            for error in parsed["protocol_errors"]
        )
    )


def test_discovery_agent_enforces_total_simulator_episode_budget():
    executor = _LinearNoisyExecutor(seed=3)
    first = '<run_experiment>[{"x":1.0},{"x":2.0},{"x":3.0}]</run_experiment>'
    final = "\n<final_law>\ndef discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n    return pos2, velocity2\n</final_law>\n<explanation>A bounded synthetic response.</explanation>\n"
    replies = iter([first, final])
    agent = DiscoveryAgent(
        model="fake",
        executor=executor,
        verbose=False,
        max_rounds=4,
        min_rounds=2,
        max_simulator_episodes=2,
    )
    with patch(
        "scienceagent.llm_client.complete", side_effect=lambda **kwargs: next(replies)
    ):
        law = agent.run()
    assert law is not None
    assert agent.simulator_episodes_used == 2
    assert agent.remaining_simulator_episodes == 0
    assert len(agent.conversation_log[0]["experiment_input"]) == 2
    assert agent.conversation_log[0]["simulator_episodes_used"] == 2
    assert agent.conversation_log[1]["action"] == "final_law"
