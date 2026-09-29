import copy
import hashlib
import json
import math
import re
import signal
import threading
from typing import Optional
from scienceagent import llm_client
from scienceagent.context_window import prepare_messages_for_context
from scienceagent.executor import SimulationExecutor

MAX_ROUNDS = 10
MIN_ROUNDS = 2


class _FinalReplayTimeout(BaseException):
    pass


_SYSTEM_PROMPT_PATH = "PhysicsSchool/prompts/_template_interactive.md"
_DEFAULT_LAW_STUB = "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n    return final_pos2, final_vel2\n"
_DEFAULT_EXPERIMENT_FORMAT = '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0, 2.0]}]</run_experiment>'
_MSE_FIT_PROMPT_BLOCK = '## OPTIONAL TOOL — MSE FITTING OF YOUR CANDIDATE LAW\n\nAt the end of any round, you may include a <run_mse_fit> tag ALONGSIDE your\n<run_experiment> (or by itself) to ask the system to fit your candidate\nlaw\'s free parameters against the trajectory data you have already\ncollected this run. This is the same scipy.optimize-based optimisation that\nthe evaluator runs at the end of the mission, so use it to refine your law\nmid-discovery rather than waiting until submission.\n\nRules:\n- The fit only sees data from THIS run (filtered by run_id), not other runs.\n- An <run_mse_fit> call does NOT consume a round on its own — running it\n  alongside <run_experiment> still counts as one round.\n- You may include AT MOST one <run_mse_fit> per round.\n- The body of <run_mse_fit> must contain the SAME source you would put in\n  <final_law>: a `discovered_law(...)` function and (optionally) a\n  `fit_parameters()` function declaring init values and bounds for free\n  parameters. Without a `fit_parameters()` block the system reports the\n  loss of your current hard-coded constants but cannot tune them.\n- The system replies with a <mse_fit_output> JSON block containing\n  `loss_before`, `loss_after`, `fitted_params`, `declared_params`,\n  `n_training`, `training_mode`, and `error`. When paired counterfactuals\n  exist, `training_mode="paired_conversation"` means the optimizer used\n  branch-minus-factual effects with common noise cancelled, matching the\n  evaluator. Use fitted values to calibrate a structurally justified family;\n  never let a fit override controlled scaling or sign evidence by silently\n  changing that family.\n- Do NOT submit <run_mse_fit> in your final-law round — that round must\n  contain ONLY <final_law> and <explanation>.\n- If <run_mse_fit> reports an error (compile failure, invalid\n  fit_parameters spec, or no_training_trajectories), fix the law in the\n  next round before submitting <final_law>.\n\nExample:\n<run_mse_fit>\ndef discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):\n    import numpy as np\n    alpha = params.get("alpha", 0.5)\n    G     = params.get("G", 1.0)\n    return final_pos2, final_vel2\n\ndef fit_parameters():\n    return {\n        "alpha": {"init": 0.5, "bounds": [0.1, 1.5]},\n        "G":     {"init": 1.0, "bounds": [0.01, 10.0]},\n    }\n</run_mse_fit>\n'


def _load_system_prompt(prompt_path: str = None, instructions_path: str = None) -> str:
    import os

    prompt_path = prompt_path or _SYSTEM_PROMPT_PATH
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", ".."))

    def _read(rel: str) -> str:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            full = rel
        with open(full) as f:
            return f.read()

    template = _read(prompt_path)
    if "{{world_instructions}}" in template:
        if not instructions_path:
            raise ValueError(
                f"Prompt {prompt_path!r} contains a {{world_instructions}} placeholder but no instructions_path was supplied."
            )
        template = template.replace(
            "{{world_instructions}}", _read(instructions_path).strip()
        )
    return template


class DiscoveryAgent:
    def __init__(
        self,
        model: str,
        executor,
        mission: Optional[str] = None,
        max_tokens: int = 4096,
        verbose: bool = True,
        show_experiment_output: bool = False,
        system_prompt_path: str = None,
        instructions_path: str = None,
        law_stub: str = None,
        experiment_format: str = None,
        max_rounds: int = MAX_ROUNDS,
        min_rounds: int = MIN_ROUNDS,
        trajectory_logger=None,
        scm_pipeline=None,
        max_simulator_episodes: Optional[int] = None,
        context_max_chars: Optional[int] = None,
    ):
        self.model = model
        self.executor = executor
        self.mission = mission
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.show_experiment_output = show_experiment_output
        self._system_prompt_path = system_prompt_path or _SYSTEM_PROMPT_PATH
        self._instructions_path = (
            instructions_path or "PhysicsSchool/prompts/2particle_instructions.md"
        )
        self._law_stub = law_stub or _DEFAULT_LAW_STUB
        self._experiment_format = experiment_format or _DEFAULT_EXPERIMENT_FORMAT
        self.max_rounds = max(1, int(max_rounds))
        self.min_rounds = max(1, min(int(min_rounds), self.max_rounds))
        self.trajectory_logger = trajectory_logger
        self.scm_pipeline = scm_pipeline
        self.max_simulator_episodes = (
            None
            if max_simulator_episodes is None
            else max(0, int(max_simulator_episodes))
        )
        self.context_max_chars = (
            None if context_max_chars is None else max(1, int(context_max_chars))
        )
        self.simulator_episodes_used = 0
        self._system = self._build_system_prompt()
        self.conversation_log: list[dict] = []
        self.discovered_explanation: Optional[str] = None
        self.context_compaction_log: list[dict] = []

    def run(self) -> Optional[str]:
        self.conversation_log = []
        self.discovered_explanation = None
        self.simulator_episodes_used = 0
        self.context_compaction_log = []
        if self.scm_pipeline is not None:
            self.scm_pipeline.reset()
        messages = []
        if self.mission:
            messages.append({"role": "user", "content": self.mission})
        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n{'=' * 52}")
                print(
                    f"\U000f0668 \U000f09d1  science agent experimenting at round {round_num}/{self.max_rounds} \U000f09d1  \U000f0668"
                )
                print(f"{'=' * 52}")
            round_entry = {
                "round": round_num,
                "system_message": None,
                "llm_reply": None,
                "action": None,
                "experiment_input": None,
                "experiment_output": None,
                "experiment_error": None,
                "final_law": None,
                "explanation": None,
                "raw_explanation": None,
                "mse_fit_input": None,
                "mse_fit_output": None,
                "scm": None,
                "simulator_episodes_before": self.simulator_episodes_used,
                "simulator_episodes_used": 0,
                "simulator_episodes_after": self.simulator_episodes_used,
            }
            if self.scm_pipeline is not None:
                scm_context = self.scm_pipeline.round_context(round_num)
                messages.append({"role": "user", "content": scm_context})
                round_entry["system_message"] = _join_sys(
                    round_entry["system_message"], scm_context
                )
            remaining_episodes = self.remaining_simulator_episodes
            effective_remaining_episodes = (
                remaining_episodes
                if self.scm_pipeline is not None
                else remaining_episodes
            )
            if self.max_simulator_episodes is not None:
                budget_msg = f"<simulator_budget>total={self.max_simulator_episodes},used={self.simulator_episodes_used},remaining={remaining_episodes},available={effective_remaining_episodes}</simulator_budget>\nEvery factual experiment and every counterfactual branch costs one simulator episode. Propose only actions that fit the remaining budget."
                messages.append({"role": "user", "content": budget_msg})
                round_entry["system_message"] = _join_sys(
                    round_entry["system_message"], budget_msg
                )
            if self.max_rounds >= 2 and round_num == self.max_rounds - 1:
                final_requirement = (
                    "<final_law>, <explanation>, and <scm_final>"
                    if self.scm_pipeline is not None
                    else "<final_law>"
                )
                warn_msg = f"Warning: this is round {round_num} of {self.max_rounds}. You have one round remaining. Your next response MUST submit {final_requirement} — you will not be able to run further experiments after this."
                messages.append({"role": "user", "content": warn_msg})
                round_entry["system_message"] = _join_sys(
                    round_entry["system_message"], warn_msg
                )
            budget_exhausted = self.remaining_simulator_episodes == 0
            if round_num == self.max_rounds or budget_exhausted or False:
                scm_final_stub = ""
                if self.scm_pipeline is not None:
                    scm_final_stub = '\n<explanation>Your concise physical explanation.</explanation>\n<scm_final>\n{"selected_candidate_id":"your best candidate","remaining_alternatives":[],"identified_mechanism":"your causal mechanism","operator_or_symmetry":"operator or measured symmetry","source_response_roles":"source and response roles","scalar_magnitude_law":"measured scalar law","vector_law":"signed vector equation","time_and_scale_regimes":"time dependence and near/far regimes","remaining_uncertainty":"what remains uncertain","evidence_summary":"decisive interventions and counterexamples","noise_cancelled_scaling_check":"controlled input exponents, local radius slopes, sign tests, and clock response","operator_correspondence_check":"named operator or Green-function correspondence supported by those invariances","evidence_consistency_check":"confirm code and explanation preserve the selected candidate defining features","numerical_robustness_check":"near-collision regularization, integrator success, and collected-trajectory code check"}\n</scm_final>'
                force_reason = "This is your final round."
                if self.scm_pipeline is None:
                    force_msg = (
                        force_reason
                        + " You MUST now submit your best guess as a <final_law> regardless of confidence. Do not run more experiments.\n\n<final_law>\n"
                        + self._law_stub
                        + "</final_law>\n<explanation>Your physical explanation.</explanation>"
                    )
                else:
                    hypothesis_only = bool(
                        getattr(self.scm_pipeline, "textual_hypotheses_only", False)
                    )
                    representation_audit = (
                        "Before answering, audit the selected free-form text hypothesis against the collected evidence. The <scm_final> tag is a legacy final-summary envelope, not a request to build an SCM. "
                        if hypothesis_only
                        else "Before answering, audit the selected SCM against every required <scm_final> field. "
                    )
                    force_msg = (
                        force_reason
                        + " You MUST now submit your best guess as a <final_law> regardless of confidence. Do not run more experiments.\n\n"
                        + representation_audit
                        + "Your <explanation> and executable law must preserve any supported operator, non-locality, time dependence, scale crossover, sign rule, or latent class; do not replace it with a generic fit. Reconcile the law explicitly with the runtime's noise-cancelled causal response table, controlled input exponents, ordered local radius slopes, sign relations, and absolute-time responses. Name the physical operator or Green-function correspondence supported by those invariances.\n\n<final_law>\n"
                        + self._law_stub
                        + "</final_law>"
                        + scm_final_stub
                    )
                messages.append({"role": "user", "content": force_msg})
                round_entry["system_message"] = _join_sys(
                    round_entry["system_message"], force_msg
                )
            reply = self._complete(messages)
            round_entry["llm_reply"] = reply
            if self.verbose:
                print(f"\n[Science Agent]\n{reply}")
            messages.append({"role": "assistant", "content": reply})
            if self.scm_pipeline is not None:
                scm_round = self.scm_pipeline.ingest_reply(
                    reply, round_num, max_episode_cost=self.remaining_simulator_episodes
                )
                round_entry["scm"] = scm_round
                if scm_round["protocol_errors"]:
                    protocol_error_msg = (
                        "<scm_protocol_errors>\n"
                        + json.dumps(
                            scm_round["protocol_errors"], separators=(",", ":")
                        )
                        + "\n</scm_protocol_errors>"
                    )
                    messages.append({"role": "user", "content": protocol_error_msg})
            final_law = _extract_tag(reply, "final_law")
            if final_law is not None:
                audit_blockers = []
                can_continue_auditing = (
                    round_num < self.max_rounds
                    and self.remaining_simulator_episodes != 0
                )
                if (
                    self.scm_pipeline is not None
                    and self.scm_pipeline.strict
                    and can_continue_auditing
                ):
                    audit_blockers = self.scm_pipeline.finalization_blockers()
                if audit_blockers:
                    self.scm_pipeline.final_claim = None
                    warning = (
                        "Your final submission is premature because these generic causal audits are still unresolved: "
                        + "; ".join(audit_blockers)
                        + ". Continue with a matched counterfactual experiment. Combine missing branches in one design when the episode budget is tight; do not resubmit <final_law> until the runtime reports no mandatory_audit_blockers."
                    )
                    round_entry["action"] = "warning"
                    round_entry["rejected_final_law"] = final_law.strip()
                    round_entry["finalization_blockers"] = audit_blockers
                    round_entry["system_message"] = _join_sys(
                        round_entry["system_message"], warning
                    )
                    self.conversation_log.append(round_entry)
                    messages.append({"role": "user", "content": warning})
                    continue
                if round_num >= self.min_rounds or False:
                    if self.verbose:
                        print("\n[Agent submitted final law]")
                    explanation = _extract_tag(reply, "explanation")
                    if explanation is None:
                        if self.verbose:
                            print(
                                "[Warning] Final law submitted without <explanation>. Re-prompting once."
                            )
                        followup_msg = "Your <final_law> submission is accepted, but you did not include the required <explanation> tag. Reply NOW with ONLY a single <explanation>...</explanation> block containing a 2–3 sentence plain-English description of the physical system you discovered. Do not include any other text or tags."
                        messages.append({"role": "user", "content": followup_msg})
                        followup_reply = self._complete(messages)
                        messages.append(
                            {"role": "assistant", "content": followup_reply}
                        )
                        if self.verbose:
                            print(
                                f"\n[Science Agent — explanation follow-up]\n{followup_reply}"
                            )
                        explanation = _extract_tag(followup_reply, "explanation")
                    raw_explanation = explanation.strip() if explanation else None
                    final_law, repaired_explanation = self._repair_final_law_if_needed(
                        round_num=round_num,
                        law_source=final_law.strip(),
                        raw_explanation=raw_explanation,
                        messages=messages,
                        round_entry=round_entry,
                    )
                    final_law, reflected_explanation = (
                        final_law,
                        repaired_explanation
                        if repaired_explanation is not None
                        else raw_explanation,
                    )
                    if repaired_explanation is not None:
                        raw_explanation = repaired_explanation
                    if reflected_explanation is not None:
                        raw_explanation = reflected_explanation
                    self.discovered_explanation = self._audited_explanation(
                        raw_explanation
                    )
                    round_entry["action"] = "final_law"
                    round_entry["final_law"] = final_law.strip()
                    round_entry["raw_explanation"] = raw_explanation
                    round_entry["explanation"] = self.discovered_explanation
                    if self.scm_pipeline is not None:
                        self.scm_pipeline.finalize(
                            law_source=final_law.strip(),
                            explanation=self.discovered_explanation,
                            reply=reply,
                        )
                        round_entry[
                            "scm_final_state"
                        ] = self.scm_pipeline.state_summary()
                    self.conversation_log.append(round_entry)
                    return final_law.strip()
                else:
                    warn = f"You have only run {round_num} round(s) of experiments. You must run at least {self.min_rounds} rounds before submitting a final law. Please design and run at least one more experiment."
                    if self.verbose:
                        print(
                            f"[Warning] Agent submitted final law too early (round {round_num}/{self.min_rounds} minimum). Requiring more experiments."
                        )
                    round_entry["action"] = "warning"
                    round_entry["system_message"] = warn
                    self.conversation_log.append(round_entry)
                    messages.append({"role": "user", "content": warn})
                    continue
            experiment_block = _extract_tag(reply, "run_experiment")
            mse_fit_block = _extract_tag(reply, "run_mse_fit")
            validation_content = None
            scm_selection = (
                self.scm_pipeline.pending_selection
                if self.scm_pipeline is not None
                else None
            )
            if (
                experiment_block is None
                and mse_fit_block is None
                and (scm_selection is None)
            ):
                if self.verbose:
                    print(
                        "[Warning] No recognized tag in response. Prompting agent to continue."
                    )
                if self.scm_pipeline is not None:
                    no_tag_msg = (
                        "ERROR: No <design_experiments>, <run_experiment>, <run_mse_fit>, or <final_law> action was found. In SCM mode, normally submit <scm_update> followed by <design_experiments> as specified in the structured causal-discovery protocol. You may also use the legacy experiment action below:\n\n"
                        + self._experiment_format
                    )
                else:
                    no_tag_msg = (
                        "ERROR: No <run_experiment>, <run_mse_fit>, or <final_law> tag found in your response. You must respond with one of these XML tags — no code fences, no markdown, just the raw tag.\n\nOption 1 — run an experiment:\n"
                        + self._experiment_format
                        + "\n\nOption 2 — submit your final law:\n<final_law>\n"
                        + self._law_stub
                        + "</final_law>\n\nOption 3 — fit your candidate law's free parameters against the data you have collected so far (returns MSE before/after and fitted params):\n<run_mse_fit>\n"
                        + self._law_stub
                        + "</run_mse_fit>\n\nRespond with the XML tag NOW. No explanation before or after."
                    )
                round_entry["action"] = "no_tag"
                round_entry["system_message"] = _join_sys(
                    round_entry["system_message"], no_tag_msg
                )
                self.conversation_log.append(round_entry)
                messages.append({"role": "user", "content": no_tag_msg})
                continue
            if scm_selection is not None:
                experiment_block = None
                try:
                    episode_cost = self.scm_pipeline.pending_episode_cost()
                    self._reserve_simulator_episodes(episode_cost, round_entry)
                    execution = self.scm_pipeline.execute_pending(self.executor)
                    output_content = (
                        f"<{execution['display_tag']}>\n"
                        + _compact_json(execution["outcome"])
                        + f"\n</{execution['display_tag']}>"
                    )
                    round_entry["action"] = (
                        "counterfactual"
                        if execution["kind"] == "counterfactual"
                        else "scm_experiment"
                    )
                    round_entry["experiment_input"] = execution["experiment_input"]
                    round_entry["experiment_output"] = execution["experiment_output"]
                    log_pairs = execution["log_pairs"]
                    self._log_trajectories(
                        round_num,
                        round_entry["action"],
                        [pair[0] for pair in log_pairs],
                        [pair[1] for pair in log_pairs],
                    )
                    validation = self.scm_pipeline.validate_pending(execution)
                    round_entry["scm"]["validation"] = validation
                    validation_content = self.scm_pipeline.validation_feedback(
                        validation
                    )
                    messages.append({"role": "user", "content": output_content})
                    messages.append({"role": "user", "content": validation_content})
                except Exception as e:
                    output_content = f"<experiment_output>\nError running SCM-selected experiment: {e}\n</experiment_output>"
                    round_entry["action"] = "scm_experiment"
                    round_entry["experiment_error"] = str(e)
                    failure = self.scm_pipeline.record_execution_error(e)
                    round_entry["scm"]["execution_failure"] = failure
                    messages.append({"role": "user", "content": output_content})
                if self.verbose and self.show_experiment_output:
                    print(f"\n[Simulator — SCM-selected]\n{output_content}")
            if experiment_block is not None:
                try:
                    exp_input = json.loads(experiment_block)
                    if not isinstance(exp_input, list) or not exp_input:
                        raise ValueError(
                            "<run_experiment> must contain a non-empty JSON list"
                        )
                    requested = len(exp_input)
                    allowed = self._allowed_simulator_episodes(requested)
                    if allowed <= 0:
                        raise RuntimeError("simulator episode budget exhausted")
                    if allowed < requested:
                        exp_input = exp_input[:allowed]
                        truncation_msg = f"Simulator budget allowed {allowed} of {requested} requested experiments; only that prefix was executed."
                        round_entry["system_message"] = _join_sys(
                            round_entry["system_message"], truncation_msg
                        )
                    self._reserve_simulator_episodes(len(exp_input), round_entry)
                    results = self.executor.run(exp_input)
                    output_content = (
                        "<experiment_output>\n"
                        + _compact_json(results)
                        + "\n</experiment_output>"
                    )
                    round_entry["action"] = "experiment"
                    round_entry["experiment_input"] = exp_input
                    round_entry["experiment_output"] = results
                    self._log_trajectories(round_num, "agent", exp_input, results)
                    if self.scm_pipeline is not None:
                        validation = self.scm_pipeline.observe_external_intervention(
                            exp_input, results
                        )
                        round_entry["scm"]["validation"] = validation
                        validation_content = self.scm_pipeline.validation_feedback(
                            validation
                        )
                except Exception as e:
                    output_content = f"<experiment_output>\nError running experiment: {e}\n</experiment_output>"
                    round_entry["action"] = "experiment"
                    round_entry["experiment_error"] = str(e)
                if self.verbose and self.show_experiment_output:
                    print(f"\n[Simulator]\n{output_content}")
                messages.append({"role": "user", "content": output_content})
                if self.scm_pipeline is not None and validation_content is not None:
                    messages.append({"role": "user", "content": validation_content})
            if mse_fit_block is not None:
                fit_output_content = self._run_mse_fit(
                    round_num, mse_fit_block, round_entry
                )
                if self.scm_pipeline is not None:
                    self.scm_pipeline.record_mse_fit(
                        round_num, round_entry.get("mse_fit_output")
                    )
                if self.verbose and self.show_experiment_output:
                    print(f"\n[Fitter]\n{fit_output_content}")
                messages.append({"role": "user", "content": fit_output_content})
                if round_entry["action"] is None:
                    round_entry["action"] = "mse_fit"
            self.conversation_log.append(round_entry)
        if self.verbose:
            print(
                f"\n[Agent did not submit a final law within {self.max_rounds} rounds]"
            )
        return None

    def _complete(self, messages: list[dict]) -> str:
        context_max_chars = getattr(self, "context_max_chars", None)
        prepared, report = prepare_messages_for_context(
            messages, system=self._system, max_chars=context_max_chars
        )
        if context_max_chars is not None:
            if not hasattr(self, "context_compaction_log"):
                self.context_compaction_log = []
            report["call_index"] = len(self.context_compaction_log) + 1
            self.context_compaction_log.append(report)
        return llm_client.complete(
            model=self.model,
            messages=prepared,
            system=self._system,
            max_tokens=self.max_tokens,
        )

    @property
    def remaining_simulator_episodes(self) -> Optional[int]:
        if self.max_simulator_episodes is None:
            return None
        return max(self.max_simulator_episodes - self.simulator_episodes_used, 0)

    def _allowed_simulator_episodes(self, requested: int) -> int:
        requested = max(0, int(requested))
        remaining = self.remaining_simulator_episodes
        if self.scm_pipeline is not None:
            remaining = remaining
        return requested if remaining is None else min(requested, remaining)

    def _reserve_simulator_episodes(
        self, requested: int, round_entry: Optional[dict] = None
    ) -> int:
        allowed = self._allowed_simulator_episodes(requested)
        if allowed != requested:
            raise RuntimeError(
                f"simulator action costs {requested} episode(s), but only {allowed} remain"
            )
        self.simulator_episodes_used += allowed
        if round_entry is not None:
            round_entry["simulator_episodes_used"] = (
                int(round_entry.get("simulator_episodes_used") or 0) + allowed
            )
            round_entry["simulator_episodes_after"] = self.simulator_episodes_used
        return allowed

    def _run_mse_fit(self, round_num: int, law_source: str, round_entry: dict) -> str:
        round_entry["mse_fit_input"] = law_source.strip()
        if self.trajectory_logger is None:
            result = {
                "error": "trajectory CSV logging is disabled, so no data is available for MSE fitting in this run.",
                "loss_before": None,
                "loss_after": None,
                "fitted_params": {},
                "declared_params": {},
                "n_training": 0,
                "training_mode": None,
            }
            round_entry["mse_fit_output"] = result
            return (
                "<mse_fit_output>\n"
                + json.dumps(result, separators=(",", ":"))
                + "\n</mse_fit_output>"
            )
        from scienceagent.evaluator import _extract_training_trajectories
        from scienceagent.mse_fitting import fit_law, redact_world_leak

        try:
            live_training = _extract_training_trajectories(
                [*self.conversation_log, round_entry]
            )
            result = fit_law(
                law_source=law_source,
                world=self.trajectory_logger.world,
                csv_path=self.trajectory_logger.csv_path,
                run_id=self.trajectory_logger.run_id,
                training_samples=live_training or None,
            )
        except Exception as e:
            result = {
                "error": "unexpected_error: "
                + redact_world_leak(
                    e, self.trajectory_logger.csv_path, self.trajectory_logger.world
                ),
                "loss_before": None,
                "loss_after": None,
                "fitted_params": {},
                "declared_params": {},
                "n_training": 0,
                "training_mode": None,
            }
        round_entry["mse_fit_output"] = result
        if self.verbose:
            err = result.get("error")
            if err:
                print(f"[mse_fit] error: {err}")
            else:
                lb = result.get("loss_before")
                la = result.get("loss_after")
                fp = result.get("fitted_params") or {}
                pretty = (
                    ", ".join((f"{k}={v:.4g}" for k, v in fp.items()))
                    or "(no fit_parameters)"
                )
                lb_str = f"{lb:.4g}" if isinstance(lb, (int, float)) else "n/a"
                la_str = f"{la:.4g}" if isinstance(la, (int, float)) else "n/a"
                print(f"[mse_fit] loss {lb_str} -> {la_str}  fitted: {pretty}")
        return (
            "<mse_fit_output>\n"
            + json.dumps(result, separators=(",", ":"))
            + "\n</mse_fit_output>"
        )

    def _evaluate_final_law_consistency(self, round_num: int, law_source: str) -> dict:
        scratch: dict = {}
        self._run_mse_fit(round_num, law_source, scratch)
        fit_result = scratch.get("mse_fit_output") or {}
        fitted_params = (
            fit_result.get("fitted_params") if isinstance(fit_result, dict) else {}
        ) or {}
        causal_check = (
            self.scm_pipeline.executable_causal_check(
                law_source, fitted_params=fitted_params
            )
            if self.scm_pipeline is not None
            else {
                "available": False,
                "passed": True,
                "reason": "structured SCM runtime is disabled",
            }
        )
        fit_error = (
            fit_result.get("error")
            if isinstance(fit_result, dict)
            else "invalid automatic fit result"
        )
        unavailable_fit_prefixes = (
            "trajectory CSV logging is disabled",
            "trajectory CSV not found for this run",
            "run_id is required",
            "no training trajectories logged yet",
            "no usable training samples after CSV reshape",
            "failed to reshape CSV into training samples",
            "world '",
        )
        fit_contract_error = bool(fit_error) and (
            not str(fit_error).startswith(unavailable_fit_prefixes)
        )
        if fit_contract_error:
            causal_check = dict(causal_check)
            causal_check["causal_replay_passed"] = bool(causal_check.get("passed"))
            causal_check["available"] = True
            causal_check["passed"] = False
            causal_check[
                "reason"
            ] = "final source has an invalid fit_parameters/compile contract: " + str(
                fit_error
            )
        training_check = {
            "available": False,
            "passed": True,
            "loss_after": None,
            "loss_relative_to_noise_variance": None,
        }
        loss_after = (
            fit_result.get("loss_after") if isinstance(fit_result, dict) else None
        )
        noise_std = (
            float(self.scm_pipeline.observation_noise_std)
            if self.scm_pipeline is not None
            else 0.0
        )
        if (
            isinstance(loss_after, (int, float))
            and (not isinstance(loss_after, bool))
            and math.isfinite(float(loss_after))
            and (noise_std > 0.0)
        ):
            relative_loss = float(loss_after) / max(noise_std**2, 1e-12)
            training_check = {
                "available": True,
                "passed": bool(relative_loss <= 1.25),
                "loss_after": float(loss_after),
                "loss_relative_to_noise_variance": relative_loss,
                "threshold": 1.25,
            }
            if not causal_check.get("available") and (not training_check["passed"]):
                causal_check = dict(causal_check)
                causal_check["available"] = True
                causal_check["passed"] = False
                causal_check[
                    "reason"
                ] = "final executable replay loss on already observed trajectories is too high relative to observation noise"
        semigroup_check = {
            "available": False,
            "passed": True,
            "reason": "semigroup consistency gate is disabled",
            "simulator_episodes_spent": 0,
        }
        return {
            "fit": fit_result,
            "causal_check": causal_check,
            "training_check": training_check,
            "semigroup_check": semigroup_check,
        }

    def _evaluate_final_law_consistency_bounded(
        self, round_num: int, law_source: str, *, timeout_seconds: Optional[float]
    ) -> dict:
        if (
            timeout_seconds is None
            or timeout_seconds <= 0.0
            or threading.current_thread() is not threading.main_thread()
            or (not hasattr(signal, "setitimer"))
        ):
            return self._evaluate_final_law_consistency(round_num, law_source)
        timeout = float(timeout_seconds)
        old_handler = signal.getsignal(signal.SIGALRM)

        def _raise_timeout(_signum, _frame):
            raise _FinalReplayTimeout()

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return self._evaluate_final_law_consistency(round_num, law_source)
        except _FinalReplayTimeout:
            return {
                "fit": {
                    "error": f"runtime_timeout_after_{timeout:g}_seconds",
                    "loss_before": None,
                    "loss_after": None,
                    "fitted_params": {},
                    "declared_params": {},
                    "n_training": 0,
                    "training_mode": "bounded_public_replay",
                },
                "causal_check": {
                    "available": True,
                    "passed": False,
                    "reason": "untrusted final executable exceeded the bounded public-replay wall-clock limit",
                    "runtime_timeout_seconds": timeout,
                },
                "training_check": {
                    "available": True,
                    "passed": False,
                    "loss_after": None,
                    "loss_relative_to_noise_variance": None,
                },
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

    @staticmethod
    def _final_check_rank(record: dict) -> tuple:
        check = record.get("causal_check") or {}
        fit = record.get("fit") or {}
        trusted_replay = record.get("trusted_finite_replay") or {}
        fit_valid = bool(trusted_replay.get("passed")) or not bool(
            fit.get("error") if isinstance(fit, dict) else True
        )
        median_error = check.get("median_relative_vector_error")
        p80_error = check.get("p80_relative_vector_error")
        training_check = record.get("training_check") or {}
        relative_training_loss = training_check.get("loss_relative_to_noise_variance")
        return (
            DiscoveryAgent._final_check_is_finite(record),
            bool(check.get("available") and check.get("passed")),
            bool(trusted_replay.get("available") and trusted_replay.get("passed")),
            bool(check.get("passed")),
            fit_valid,
            -float(median_error)
            if isinstance(median_error, (int, float))
            and math.isfinite(float(median_error))
            else float("-inf"),
            -float(p80_error)
            if isinstance(p80_error, (int, float)) and math.isfinite(float(p80_error))
            else float("-inf"),
            -float(relative_training_loss)
            if isinstance(relative_training_loss, (int, float))
            and math.isfinite(float(relative_training_loss))
            else float("-inf"),
        )

    @staticmethod
    def _final_training_loss(record: dict) -> float:
        fit = record.get("fit") or {}
        value = fit.get("loss_after")
        if (
            isinstance(value, (int, float))
            and (not isinstance(value, bool))
            and math.isfinite(float(value))
        ):
            return float(value)
        return float("inf")

    @staticmethod
    def _final_check_is_finite(record: dict) -> bool:
        semigroup_check = record.get("semigroup_check") or {}
        if semigroup_check.get("available") and (not semigroup_check.get("passed")):
            return False
        trusted_replay = record.get("trusted_finite_replay") or {}
        if trusted_replay.get("available"):
            return bool(trusted_replay.get("passed"))
        fit = record.get("fit") or {}
        if not isinstance(fit, dict) or fit.get("error"):
            return False
        loss = fit.get("loss_after")
        if not (
            isinstance(loss, (int, float))
            and (not isinstance(loss, bool))
            and math.isfinite(float(loss))
        ):
            return False
        check = record.get("causal_check") or {}
        return bool(check.get("passed"))

    def _repair_final_law_if_needed(
        self,
        *,
        round_num: int,
        law_source: str,
        raw_explanation: Optional[str],
        messages: list[dict],
        round_entry: dict,
    ) -> tuple[str, Optional[str]]:
        if self.scm_pipeline is None:
            return (law_source, raw_explanation)
        require_finite = False
        replay_timeout = None
        initial = self._evaluate_final_law_consistency_bounded(
            round_num, law_source, timeout_seconds=replay_timeout
        )
        checks = [{"attempt": 0, "source": "initial_submission", **initial}]
        round_entry["final_code_checks"] = checks
        round_entry["final_code_repaired"] = False
        round_entry["final_repair_replies"] = []
        initial_check = initial.get("causal_check") or {}
        initial_causal_pass = bool(initial_check.get("passed"))
        initial_accept = bool(
            initial_causal_pass
            and (not require_finite or self._final_check_is_finite(initial))
        )
        time_alias_compiler = getattr(
            self.scm_pipeline, "deductive_time_modulated_laws", None
        )
        time_deductions = time_alias_compiler() if callable(time_alias_compiler) else []
        time_deductions = [
            deduction
            for deduction in time_deductions
            if isinstance(deduction, dict) and isinstance(deduction.get("source"), str)
        ]
        ambiguous_time_aliases = len(time_deductions) > 1
        if not ambiguous_time_aliases and (
            not require_finite
            and (not initial_check.get("available"))
            or initial_accept
        ):
            return (law_source, raw_explanation)
        best_law = law_source
        best_explanation = raw_explanation
        best_check = initial
        repair_context = initial
        deduction_records = []
        compilation_specs: list[tuple[str, dict]] = []
        if not initial_causal_pass:
            compact_compiler = getattr(
                self.scm_pipeline, "deductive_compactified_law", None
            )
            compact_deduction = (
                compact_compiler() if callable(compact_compiler) else None
            )
            if isinstance(compact_deduction, dict) and isinstance(
                compact_deduction.get("source"), str
            ):
                compilation_specs.append(
                    ("deductive_compactified_law", compact_deduction)
                )
        if time_deductions and (ambiguous_time_aliases or not initial_causal_pass):
            compilation_specs.extend(
                (
                    ("deductive_time_modulated_laws", deduction)
                    for deduction in time_deductions
                )
            )
        time_alias_evaluations: list[tuple[dict, str, dict]] = []
        for compiler_name, deduction in compilation_specs:
            deduced_law = deduction["source"].strip()
            deduced_check = self._evaluate_final_law_consistency_bounded(
                round_num, deduced_law, timeout_seconds=replay_timeout
            )
            check_record = {
                "attempt": "deductive_compilation",
                "source": "runtime_scm_deduction",
                "compiler": compiler_name,
                "deduction": {
                    key: copy.deepcopy(value)
                    for key, value in deduction.items()
                    if key != "source"
                },
                **deduced_check,
            }
            checks.append(check_record)
            deduction_records.append(
                {
                    "compiler": compiler_name,
                    "deduction": copy.deepcopy(deduction.get("deduction")),
                }
            )
            round_entry["final_deductive_compilation"] = copy.deepcopy(
                deduction.get("deduction")
            )
            round_entry["final_deductive_compilations"] = copy.deepcopy(
                deduction_records
            )
            if (
                compiler_name == "deductive_time_modulated_laws"
                and ambiguous_time_aliases
            ):
                time_alias_evaluations.append((deduction, deduced_law, deduced_check))
            elif self._final_check_rank(deduced_check) > self._final_check_rank(
                best_check
            ):
                best_law = deduced_law
                best_check = deduced_check
                repair_context = deduced_check
        if time_alias_evaluations:
            eligible_aliases = [
                item
                for item in time_alias_evaluations
                if (item[2].get("causal_check") or {}).get("passed")
                and (item[2].get("training_check") or {}).get("passed")
                and math.isfinite(self._final_training_loss(item[2]))
            ]
            if eligible_aliases:
                selected = min(
                    eligible_aliases,
                    key=lambda item: self._final_training_loss(item[2]),
                )
                selected_deduction, selected_law, selected_check = selected
                selected_loss = self._final_training_loss(selected_check)
                initial_loss = self._final_training_loss(initial)
                select_alias = bool(
                    not initial_causal_pass
                    or not math.isfinite(initial_loss)
                    or selected_loss <= 0.995 * initial_loss
                )
                alias_rows = []
                for deduction, _, check in time_alias_evaluations:
                    alias = deduction.get("time_alias") or {}
                    alias_rows.append(
                        {
                            "alias_index": alias.get("alias_index"),
                            "angular_frequency_omega": alias.get(
                                "angular_frequency_omega"
                            ),
                            "period": alias.get("period"),
                            "phase": alias.get("phase"),
                            "training_loss": self._final_training_loss(check)
                            if math.isfinite(self._final_training_loss(check))
                            else None,
                            "causal_check_passed": bool(
                                (check.get("causal_check") or {}).get("passed")
                            ),
                        }
                    )
                if select_alias:
                    alias = selected_deduction.get("time_alias") or {}
                    selection = {
                        "selection_method": "minimum existing paired-trajectory loss among causal-compatible odd-harmonic aliases",
                        "selected_alias_index": alias.get("alias_index"),
                        "angular_frequency_omega": float(
                            alias["angular_frequency_omega"]
                        ),
                        "period": float(alias["period"]),
                        "phase": float(alias["phase"]),
                        "training_loss": selected_loss,
                        "initial_submission_training_loss": initial_loss
                        if math.isfinite(initial_loss)
                        else None,
                        "candidate_aliases": alias_rows,
                        "simulator_episodes_added": 0,
                    }
                    self.scm_pipeline.time_alias_selection = copy.deepcopy(selection)
                    round_entry["final_time_alias_selection"] = copy.deepcopy(selection)
                    best_law = selected_law
                    best_check = selected_check
                    repair_context = selected_check
                    round_entry["final_deductive_compilation"] = copy.deepcopy(
                        selected_deduction.get("deduction")
                    )
        if initial_accept:
            round_entry["final_code_repaired"] = best_law != law_source
            return (best_law, best_explanation)
        best_accept = bool(
            (best_check.get("causal_check") or {}).get("passed")
            and (not require_finite or self._final_check_is_finite(best_check))
        )
        if best_accept:
            round_entry["final_code_repaired"] = best_law != law_source
            return (best_law, best_explanation)
        audit = (
            self.scm_pipeline.state_summary().get("noise_cancelled_causal_audit") or {}
        )
        audit_view = {
            "candidate_correspondences": audit.get("candidate_correspondences"),
            "controlled_scaling": audit.get("controlled_scaling"),
            "mechanism_fits": audit.get("mechanism_fits"),
        }
        repair_replies = []
        for attempt in range(1, 3):
            causal_rows_available = bool(
                (repair_context.get("causal_check") or {}).get("case_count")
            )
            causal_guidance = (
                "In particular, a local acceleration observed at radius r is not the global power-law coefficient: use the reported control-normalized coefficient and make fit_parameters bounds straddle it. Also reconcile sign, p1/p2 roles, unit-vector conversion, absolute time, and any fitted mechanism scale. The argument roles are fixed: p1 is the exposed source control and p2 is the exposed response/inertia or pair control; never repurpose either as time. If the audit finds absolute-time dependence, extend the entry point with `start_time=0.0` after `duration` (and before `**params`) and evaluate the coupling at `start_time+t`. "
                if causal_rows_available
                else "Use the recorded trajectory-fit diagnostics to repair the actual schema-specific dynamics, parameterization, integration, and absolute-time handling. Do not replace the task with an unrelated two-particle law. "
            )
            repair_prompt = (
                "FINAL EXECUTABLE CONSISTENCY CHECK FAILED. This check used no new simulator episode and no hidden answer: it replayed your submitted code against already collected causal responses and/or observed trajectories. Repair only the executable implementation and its parameter normalization; preserve the evidence-supported SCM. "
                + causal_guidance
                + 'The repaired source MUST implement the complete evaluator entry point in the exact task contract below; a helper alone is invalid. Preserve output shapes and semantics. If numerical integration is required, prefer `solve_ivp`; if you use an explicit loop, its timestep must be at least 0.01 and strictly greater than 0.005. If parameters are fittable, `fit_parameters` MUST be a callable function (`def fit_parameters(): return {...}`), never a dictionary variable. It may declare at most five parameters. Each entry must have the exact form `name: {"init": value, "bounds": [lower, upper]}`; do not use tuples. The discovered_law must read fitted values from its keyword arguments, not call fit_parameters() internally. Return exactly one complete <final_law>...</final_law> block and one concise <explanation>...</explanation> block. Do not request another experiment.\n\n<required_law_contract>\n'
                + self._law_stub
                + "\n</required_law_contract>\n\n<final_code_check>\n"
                + json.dumps(
                    {
                        "automatic_fit": repair_context.get("fit"),
                        "causal_check": repair_context.get("causal_check"),
                        "training_check": repair_context.get("training_check"),
                        "causal_audit": audit_view,
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n</final_code_check>"
            )
            messages.append({"role": "user", "content": repair_prompt})
            repair_reply = self._complete(messages)
            messages.append({"role": "assistant", "content": repair_reply})
            repair_replies.append(repair_reply)
            candidate_law = _extract_tag(repair_reply, "final_law")
            if candidate_law is None:
                checks.append(
                    {
                        "attempt": attempt,
                        "source": "repair_response",
                        "error": "repair response omitted <final_law>",
                    }
                )
                continue
            candidate_law = _strip_code_fences(candidate_law).strip()
            candidate_check = self._evaluate_final_law_consistency_bounded(
                round_num, candidate_law, timeout_seconds=replay_timeout
            )
            checks.append(
                {"attempt": attempt, "source": "repair_response", **candidate_check}
            )
            repair_context = candidate_check
            candidate_explanation = _extract_tag(repair_reply, "explanation")
            if self._final_check_rank(candidate_check) > self._final_check_rank(
                best_check
            ):
                best_law = candidate_law
                best_check = candidate_check
                if candidate_explanation:
                    best_explanation = candidate_explanation.strip()
            if (candidate_check.get("causal_check") or {}).get("passed") and (
                not require_finite or self._final_check_is_finite(candidate_check)
            ):
                break
        round_entry["final_repair_replies"] = repair_replies
        round_entry["final_code_repaired"] = best_law != law_source
        return (best_law, best_explanation)

    def _log_trajectories(
        self, round_num: int, source: str, exp_inputs, exp_outputs
    ) -> None:
        if self.trajectory_logger is None:
            return
        if not isinstance(exp_inputs, list) or not isinstance(exp_outputs, list):
            return
        for idx, (inp, out) in enumerate(zip(exp_inputs, exp_outputs)):
            try:
                self.trajectory_logger.log_experiment(
                    round_num=round_num,
                    source=source,
                    exp_input=inp,
                    exp_output=out,
                    exp_idx_in_round=idx,
                )
            except Exception as e:
                if self.verbose:
                    print(f"[trajectory_logger] failed to log r{round_num} e{idx}: {e}")

    def _build_system_prompt(self) -> str:
        try:
            base = _load_system_prompt(
                self._system_prompt_path, self._instructions_path
            )
        except FileNotFoundError:
            base = "You are a scientific discovery agent. Design experiments, analyze results, and discover the underlying law of physics."
        blocks = [base.rstrip()]
        if self.scm_pipeline is not None:
            blocks.append(self.scm_pipeline.prompt_block().strip())
        blocks.append(self._run_policy_note())
        return "\n\n".join(blocks)

    def _audited_explanation(self, explanation: Optional[str]) -> Optional[str]:
        if self.scm_pipeline is None:
            return explanation
        claim = (
            self.scm_pipeline.final_claim
            if isinstance(self.scm_pipeline.final_claim, dict)
            else {}
        )
        state_summary = self.scm_pipeline.state_summary()
        audit = state_summary.get("noise_cancelled_causal_audit")
        correspondences = (
            audit.get("candidate_correspondences") if isinstance(audit, dict) else None
        )
        structured_parts = []
        correspondence_text = (
            " ".join((str(item) for item in correspondences)) if correspondences else ""
        )
        metrics = state_summary.get("metrics") or {}
        support_status_known = "selected_candidate_evidence_supported" in metrics
        selected_supported = bool(metrics.get("selected_candidate_evidence_supported"))
        lower_correspondence = correspondence_text.lower()
        quantitatively_supported = any(
            (
                marker in lower_correspondence
                for marker in (
                    "quantitatively identified",
                    "quantitatively selects",
                    "full trajectories selected",
                )
            )
        )
        explicitly_weak_correspondence = any(
            (
                marker in lower_correspondence
                for marker in (
                    "compare a single extra spatial dimension",
                    "compare a screened/helmholtz",
                    "still needed to distinguish",
                    "not a unique period",
                )
            )
        )
        promote_correspondence = bool(
            correspondences
            and (
                not support_status_known
                or selected_supported
                or quantitatively_supported
                or (not explicitly_weak_correspondence)
            )
        )
        if promote_correspondence:
            structured_parts.append(
                "Runtime-derived physical conclusion from the controlled data: "
                + correspondence_text
            )
            source_roles = str(claim.get("source_response_roles") or "").strip()
            scaling = audit.get("controlled_scaling") or {}
            if not source_roles:
                p2_exponent = scaling.get("p2_power_exponent_median")
                if isinstance(p2_exponent, (int, float)) and math.isfinite(
                    float(p2_exponent)
                ):
                    if float(p2_exponent) < -0.4:
                        p2_role = "p2 is an inertial response denominator"
                    elif float(p2_exponent) > 0.4:
                        p2_role = "p2 is a multiplicative coupling magnitude"
                    else:
                        p2_role = "p2 has little resolved causal effect"
                    source_roles = (
                        "the p1=0 matched branch anchors p1 as the source; " + p2_role
                    )
            if source_roles:
                structured_parts.append("Source/response roles: " + source_roles)
            quantitative = "quantitatively selects" in correspondence_text
            if not quantitative and any(
                (
                    token in correspondence_text.lower()
                    for token in ("compactif", "screened", "crossover")
                )
            ):
                successful_fits = [
                    attempt
                    for attempt in self.scm_pipeline.fit_attempts
                    if isinstance(attempt, dict)
                    and attempt.get("error") in (None, "")
                    and isinstance(attempt.get("fitted_params"), dict)
                ]
                numeric_fits = []
                for attempt in successful_fits:
                    numeric_fits.append(
                        {
                            str(name): float(value)
                            for name, value in attempt["fitted_params"].items()
                            if isinstance(value, (int, float))
                            and (not isinstance(value, bool))
                            and math.isfinite(float(value))
                        }
                    )
                lower_correspondence = correspondence_text.lower()
                calibration = None
                if "compactif" in lower_correspondence:
                    compact_fit = next(
                        (fitted for fitted in reversed(numeric_fits) if "L" in fitted),
                        None,
                    )
                    if compact_fit:
                        length = compact_fit["L"]
                        calibration = f"compact circumference L≈{length:.4g}, hence compactification radius R=L/(2*pi)≈{length / (2.0 * math.pi):.4g}"
                elif "screened" in lower_correspondence:
                    screened_fit = next(
                        (
                            (name, value)
                            for fitted in reversed(numeric_fits)
                            for name, value in fitted.items()
                            if "lambda" in name.lower() or "screen" in name.lower()
                        ),
                        None,
                    )
                    if screened_fit:
                        calibration = f"{screened_fit[0]}≈{screened_fit[1]:.4g}"
                if calibration:
                    structured_parts.append(
                        "Correspondence-scale calibration from collected trajectories: "
                        + calibration
                    )
            time_series = scaling.get("absolute_time_response") or []
            has_three_phase_series = False
            has_sign_change = False
            clock_invariant = False
            for series in time_series:
                if not isinstance(series, list) or len(series) < 3:
                    continue
                values = [
                    float(item["inward_acceleration"])
                    for item in series
                    if isinstance(item, dict)
                    and isinstance(item.get("inward_acceleration"), (int, float))
                    and math.isfinite(float(item["inward_acceleration"]))
                ]
                if len(values) < 3:
                    continue
                has_three_phase_series = True
                has_sign_change = has_sign_change or (
                    min(values) < -1e-06 and max(values) > 1e-06
                )
                scale = max(max((abs(value) for value in values)), 1e-08)
                clock_invariant = (
                    clock_invariant or max(values) - min(values) <= 0.02 * scale
                )
            if has_sign_change:
                structured_parts.append(
                    "Absolute-time audit: at the same controlled state the force changes sign across clock phases, so the coupling is explicitly time modulated."
                )
            elif has_three_phase_series and clock_invariant:
                structured_parts.append(
                    "Absolute-time audit: matched responses at three clock phases are invariant, supporting a static interaction."
                )
            return ("Structured SCM conclusion. " + "\n".join(structured_parts)).strip()
        if str(claim.get("operator_correspondence_check") or "").strip():
            structured_parts.append(
                "Operator correspondence audit: "
                + str(claim["operator_correspondence_check"]).strip()
            )
        for label, field in (
            ("Source/response audit", "source_response_roles"),
            ("Time and scale audit", "time_and_scale_regimes"),
            ("Noise-cancelled scaling audit", "noise_cancelled_scaling_check"),
        ):
            value = str(claim.get(field) or "").strip()
            if value:
                structured_parts.append(f"{label}: {value}")
        if not structured_parts:
            return explanation
        base = (explanation or "").strip()
        appendix = "Structured SCM audit. " + "\n".join(structured_parts)
        if not base:
            return appendix
        return f"{appendix}\n\nAgent-authored synthesis: {base}".strip()

    def _run_policy_note(self) -> str:
        mse_fit_available = self.trajectory_logger is not None
        base = f"## RUN-SPECIFIC CONSTRAINTS (these override any conflicting numbers above)\n- For this session you have EXACTLY {self.max_rounds} round(s) of experiments. Plan accordingly: on round {self.max_rounds} you will be forced to submit your <final_law>.\n- You must run at least {self.min_rounds} round(s) of experiments before submitting a <final_law>.\n- If your `discovered_law` integrates the trajectory with a for-loop over some timestep `dt`, the SMALLEST value of `dt` you are allowed to use is 0.01. Do not set dt below 0.01 under any circumstances.\n- ALWAYS set `dt > 0.005` in EVERY test simulation AND in your final law — no timestep anywhere in your code may be 0.005 or smaller.\n"
        if self.max_simulator_episodes is not None:
            base += f"- You have at most {self.max_simulator_episodes} simulator episode(s) in total. Each item in an experiment list, the factual branch, and every paired-counterfactual branch costs one episode. When this budget is exhausted you must submit the final law.\n"
        if mse_fit_available:
            base += "\n" + _MSE_FIT_PROMPT_BLOCK
        return base


def _join_sys(existing: Optional[str], new: str) -> str:
    if not existing:
        return new
    return existing + "\n\n" + new


def _round_floats(obj, decimals=4):
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    return obj


def _compact_json(results) -> str:
    return json.dumps(_round_floats(results), separators=(",", ":"))


def _strip_code_fences(text: str) -> str:
    text = re.sub("```(?:xml|python|json)?\\s*\\n?", "", text)
    text = re.sub("```\\s*", "", text)
    return text


def _extract_tag(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    cleaned = _strip_code_fences(text)
    match = re.search(pattern, cleaned, re.DOTALL)
    return match.group(1) if match else None
