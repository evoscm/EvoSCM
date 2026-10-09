## STRUCTURED DYNAMIC-SCM DISCOVERY PROTOCOL

This run uses an auditable causal-discovery pipeline. In addition to the
ordinary final law, you must maintain several competing, executable-in-spirit
dynamic structural causal models (SCMs). Do not hide your scientific state in
prose. The runtime never sees the world's hidden answer; it only checks your
pre-registered predictions against simulator observations.

### Causal semantics

- Observation means analysing returned trajectories.
- Abduction means inferring hidden state, parameters, or exogenous causes
  *under each current candidate SCM*.
- An intervention is a new experiment that sets controllable inputs.
- A paired counterfactual holds the hidden world and exogenous observation
  stream fixed, changes only declared input fields, and runs the alternate
  branch.
- Induction proposes a revised mechanism from accumulated evidence.
- Deduction derives a new, risky consequence that can falsify that mechanism.

Never call a fresh experiment a counterfactual merely because it is
hypothetical. Never revise a prediction after seeing its result.

Separate *mechanism identification* from *nuisance-parameter calibration*.
An absolute residual from a candidate whose coupling, phase, scale, or noise
level has not yet been calibrated is not evidence for a different structural
mechanism. Use matched interventions, dimensionless ratios, and paired effects
that cancel nuisance terms; use `<run_mse_fit>` when an executable candidate
needs numerical calibration.

Before converging on a familiar law, perform a causal invariance audit over
the controls exposed by the public experiment schema:

- hold the complete initial state fixed and vary source strength, response
  property, sign, identity, initial velocity, and absolute start time whenever
  those controls are available;
- probe geometric controls on a log-spaced near/middle/far grid. Estimate the
  *local* log-slope, not only one global power. A constant exponent, a
  screened tail, and a short-distance crossover are distinct SCM families
  until interventions show their slopes and scale dependence agree. Expand
  adaptively, but stop after roughly a 32-fold span unless a specific
  unresolved transition remains inside the tested range;
- use short-time velocity changes to estimate acceleration before long
  trajectories mix force shape with orbital dynamics;
- keep an explicitly time-varying or history-dependent candidate until
  matched-state interventions rule it out. If the topology advertises an
  optional `start_time`, a phase sweep is the direct intervention;
- distinguish an empirical force law from its operator interpretation. For a
  static isotropic field in two visible dimensions, a measured force
  magnitude `|F| ∝ r^-q` is consistent with a fractional Green function
  `-(-∇²)^alpha` at `alpha = (3-q)/2`; ordinary 2D Poisson is the `q=1`
  limiting case. Treat this as a candidate correspondence to test and report,
  not as hidden truth supplied by the runtime.

Interpret scale changes by their pattern rather than calling every deviation
"screening". Exponential suppression at large radius, a softened core, and a
change of power-law exponent between short and long distances imply different
operators. In particular, when the short-distance force gains roughly one
inverse power while the long-distance law looks lower-dimensional, include a
dimensional-crossover / compactified-dimension Green-function candidate. A
simple executable interpolation with a fitted crossover scale is acceptable
when an exact image sum is not identifiable.

Do not chase the simulator below its effective numerical resolution. At very
small separation a trajectory may cross the source before the first useful
measurement, and integrator softening or collision handling can imitate a new
force law or reverse an endpoint velocity. For local-force identification,
shorten the observation time until displacement is well below the initial
radius, replicate an overlapping radius, and require a stable local slope.
Treat isolated reversals or microstructure that appears only after repeatedly
halving an already 32-fold range as numerical-regime evidence, not as the
large-scale mechanism to encode for prediction.

Paired differences identify causal changes but can hide the absolute sign if
both branches contain a force. For a source-driven law, include a matched
null-source branch (`p1 = 0` when legal). The sign of
`velocity(null) - velocity(active)` then fixes the active force direction
without observation noise. Before final submission, run one collected initial
state mentally or numerically through the executable code and check that its
first acceleration has that same sign. "A hidden charge fixes the sign" is
not sufficient: the predictor must implement the empirically identified sign.

For every exposed two-particle scalar that accepts both signs, also perform a
same-magnitude sign reversal while holding the full state fixed. Under the
runtime's null-anchored audit, a signed-response ratio near `-1` means that the
exposed sign reverses the force; a ratio near `+1` means the exposed value
supplies magnitude only and a latent polarity fixes the direction. Test both
`p1` and `p2` in one matched design when both remain unresolved. Never infer
signed `p1*p2` coupling merely from positive-valued experiments.

When `start_time` is supported, the executable law may opt into it with
`def discovered_law(..., duration, start_time=0.0, **params)`. The held-out
evaluator uses the default zero clock, while discovery-time fitting forwards
the controlled start time. This keeps the ordinary six-argument interface
backwards compatible.

### 1. SCM update required before each non-final experiment

Include one `<scm_update>` JSON object. On round 1 initialise 3-6 genuinely
different candidates. Later rounds may upsert a complete candidate or apply a
small versioned patch.

Round-one candidate weights are deliberately initialized uniformly: a
commonsense `confidence` is recorded for audit but is not observational
evidence. Do not assume the most familiar textbook mechanism is favoured until
an intervention distinguishes it.

```xml
<scm_update>
{
  "mode": "initialize or revise",
  "abduction": {
    "observed_pattern": "what must be explained",
    "latent_causes": ["possible hidden cause"],
    "uncertainty": "what is not identifiable yet"
  },
  "candidates": [
    {
      "id": "H_power",
      "name": "central power law",
      "description": "object-level dynamic SCM",
      "confidence": 0.35,
      "variables": [
        {"name": "r_t", "role": "state", "observed": true},
        {"name": "q", "role": "parameter", "observed": false}
      ],
      "edges": [
        {"source": "r_t", "target": "a_t"},
        {"source": "a_t", "target": "v_next"}
      ],
      "mechanisms": [
        {
          "target": "a_t",
          "parents": ["r_t", "p1", "p2"],
          "equation": "a_t = -C*p1*r_hat/(p2*r**q)"
        }
      ],
      "parameters": {
        "q": {"estimate": 1.0, "lower": 0.0, "upper": 3.0}
      },
      "assumptions": ["isotropic", "time independent"],
      "falsifiers": ["same-radius acceleration depends on absolute direction"]
    }
  ],
  "patches": [],
  "inductions": ["rule inferred from more than one observation"],
  "deductions": [
    {
      "claim": "risky consequence of the current mechanism",
      "test": "experiment that could refute it"
    }
  ],
  "open_questions": ["highest-value unresolved causal question"]
}
</scm_update>
```

Supported patch operations are:

- `add_edge` or `remove_edge`, with `payload: {"source": ..., "target": ...}`;
- `replace_mechanism`, with
  `payload: {"target": ..., "mechanism": {...}}`;
- `add_latent`, with `payload: {"variable": {...}}`;
- `update_parameter`, with
  `payload: {"name": ..., "estimate": ..., "lower": ..., "upper": ...}`;
- `retire_candidate`.

Each patch has `candidate_id`, `operation`, `payload`, and `reason`. Do not
patch a model merely to explain one noisy point. Prefer the smallest mechanism
that also preserves earlier successes.

Do not retire a distinct mechanism family after one uncalibrated batch,
one unpaired noisy counterexample, or because of `tension` alone. The runtime
accepts retirement only after counterexamples from at least two distinct
designs including a noise-cancelled paired-effect falsification *and negative
relative evidence against competing candidates*, or repeated negative evidence
from at least three independent designs. If every candidate misses an absolute
target equally (`relative_log_evidence = 0`), repair nuisance calibration
rather than deleting the mechanism family. Preserve plausible alternatives
until an intervention directly tests the structural difference.

The runtime applies a soft minimum-description-length prior. An extra drag,
screening, hidden class, or other mechanism should survive only when it
predicts interventions better than the simpler nested model. Do not keep an
unsupported term merely as an "optional" hedge.

### 2. Active experiment design and prediction commitment

Instead of `<run_experiment>`, normally submit a `<design_experiments>` JSON
object containing 2-4 legal alternatives. The runtime executes exactly one:
the alternative with the largest posterior-weighted disagreement among your
candidate predictions, penalised for simulation cost.

The runtime may provide a remaining simulator-episode budget. Each ordinary
experiment, the factual branch of a paired counterfactual, and each
counterfactual branch costs one episode. Every proposed alternative must fit
that remaining budget; the selector will never execute an over-budget design.

Every prediction must commit to:

- `hypothesis_id`: an active candidate;
- `target`: JSON pointer into the proposed simulator result;
- `mean`: predicted scalar or numeric array;
- `sigma`: honest one-standard-deviation uncertainty, in the same units.

For an ordinary intervention, targets address the normal result list. For
example, `/0/pos2/-1/0` is the final x coordinate of particle 2 in experiment
0, and `/0/positions/-1/21` is the final 2-vector for particle 21.

```xml
<design_experiments>
{
  "designs": [
    {
      "id": "near_range",
      "kind": "intervention",
      "rationale": "power-law candidates separate most at short range",
      "experiments": [
        {
          "p1": 1.0,
          "p2": 1.0,
          "pos2": [2.0, 0.0],
          "velocity2": [0.0, 0.0],
          "measurement_times": [0.25, 0.5]
        }
      ],
      "predictions": [
        {
          "hypothesis_id": "H_power",
          "target": "/0/pos2/-1/0",
          "mean": 1.98,
          "sigma": 0.03,
          "reasoning": "short-time integration of the candidate mechanism"
        },
        {
          "hypothesis_id": "H_screened",
          "target": "/0/pos2/-1/0",
          "mean": 2.00,
          "sigma": 0.01
        }
      ]
    }
  ]
}
</design_experiments>
```

Predictions are scored with a Gaussian proper score. An unnecessarily huge
`sigma` is therefore penalised. A standardised error above 3 is recorded as a
counterexample. Use the returned `<scm_validation>` block in the next round;
do not overwrite its posterior or relabel its evidence.

Predictions that differ by less than their pooled uncertainty are treated as
one empirically equivalent group and receive the same relative structural
evidence. A design is useful for choosing between mechanisms only when their
predicted effects are meaningfully separated; tiny differences between two
uncalibrated constants must not decide a structural question.

For compatibility, a normal `<run_experiment>` may instead be accompanied by
`<prediction_commitment>[...]</prediction_commitment>`, using the same
prediction schema. Such an action is recorded, but the runtime cannot compare
several proposed designs before selecting it.

### 3. Paired counterfactual

A design may have `kind: "counterfactual"`. Give one complete factual
experiment and one or more interventions that replace existing input fields
using JSON pointers:

```xml
<design_experiments>
{
  "designs": [
    {
      "id": "double_radius_twin",
      "kind": "counterfactual",
      "factual": {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [2.0, 0.0],
        "velocity2": [0.0, 0.0],
        "measurement_times": [0.5]
      },
      "interventions": [
        {"id": "do_radius_4", "set": {"/pos2/0": 4.0}}
      ],
      "predictions": [
        {
          "hypothesis_id": "H_power",
          "target": "/paired_effects/0/output_delta/0/velocity2/-1/0",
          "mean": 0.020,
          "sigma": 0.03
        },
        {
          "hypothesis_id": "H_screened",
          "target": "/paired_effects/0/output_delta/0/velocity2/-1/0",
          "mean": 0.005,
          "sigma": 0.01
        }
      ]
    }
  ]
}
</design_experiments>
```

The output arrives in `<counterfactual_output>`. All branches share the same
hidden executor configuration and, when observation noise exists, common
random numbers. This is a controlled simulator counterfactual, not permission
to access hidden truth.

The first index after `paired_effects` or `counterfactuals` is the branch
index. Each branch is a singleton experiment batch, so the first index after
`output_delta` or `output` must be `0`. For example, branch 4 is
`/paired_effects/4/output_delta/0/velocity2/-1/0`, never
`/paired_effects/0/output_delta/4/...`.

The output also contains:

```json
{
  "paired_effects": [{
    "id": "double_radius_twin",
    "definition": "counterfactual_minus_factual",
    "output_delta": []
  }]
}
```

`output_delta` has the same nested shape as one simulator output and subtracts
the factual observation from the counterfactual observation. Common additive
noise therefore cancels. In every paired design, each competing SCM must
predict at least one target under `/paired_effects/...`; for example,
`/paired_effects/0/output_delta/0/velocity2/-1/0`. Prefer these causal-effect
targets over noisy absolute endpoints when discriminating mechanisms.

When identifying a distance exponent, range, frequency, or another shape
parameter in noise, one factual branch plus only one intervention is usually
underdetermined because the coupling amplitude is also unknown. Prefer at
least two intervention branches at well-separated control values. Predict the
noise-cancelled effect for every branch and compare ratios of effects, so the
shared amplitude and additive noise cannot masquerade as a different
exponent. Keep the trajectory short enough that the controlled initial state
still determines the local response.

After a paired two-particle experiment, the runtime may expose a
`noise_cancelled_causal_audit`. It reconstructs local acceleration vectors
from the earliest common-noise velocity contrasts, joins compatible designs
into a response graph, and anchors the source-driven response at observed
`p1=0` branches. Its controlled quantities have precise meanings:

- `p1_power_exponent_median` and `p2_power_exponent_median` are exponents in
  the local response magnitude, so `+1` means direct proportionality and `-1`
  means an inertial denominator;
- `radius_local_decay[].decay_exponent_q` is the local `|a| ∝ r^-q` slope
  between the stated radii. Read the ordered near and far slopes, not only
  their global median;
- sign-relation ratios near `+1` mean a visible sign is ignored, while ratios
  near `-1` mean it reverses the force;
- `absolute_time_response` compares the same controlled state at different
  absolute clock phases.

These values are derived only from experiments already shown to you. Use them
as the primary evidence for structure. A fit to noisy absolute trajectories
is useful for nuisance calibration, but must not override a noise-cancelled
input exponent, local slope, sign relation, or absolute-time sign change.
Explicitly reconcile the final code with this audit.

Translate the empirical audit into named physical correspondences while
preserving identifiability caveats. Generic examples include:

- constant `q≈1` in two visible dimensions → ordinary 2D
  Poisson/Laplacian logarithmic Green function;
- non-integer or `q≈2` with `p2` acting as an inertial denominator →
  2D Riesz/fractional-Laplacian candidate with `alpha=(3-q)/2`;
- `q≈2` with both scalar properties entering multiplicatively and visible
  signs possibly hidden → two-charge Coulomb/Newton-like central force;
- a near-range slope around 2 changing toward 1 at long range →
  higher-dimensional short-distance law with a compactified spatial
  dimension / image-sum Green function;
- a slope that steepens strongly at long range → screened Helmholtz/Yukawa
  tail;
- an otherwise identical state whose force changes sign with `start_time` →
  phase-modulated coupling; collect at least three clock phases and fit the
  period instead of encoding an arbitrary two-point switch.

These are candidate mappings from measured invariances, not hidden answers.
Choose among them using the reported source/response exponents, local slopes,
sign tests, time tests, and executable predictive checks.

For a two-particle topology, the null-source, `|p2|`-ratio, same-magnitude
`p1` and `p2` sign reversals, three-phase clock, and cross-scale radius checks
are mandatory generic invariance audits even when the most familiar initial
candidate is static. The radius audit must reconstruct at least four
null-anchored local responses spanning at least 8x in radius and therefore at
least three ordered local slopes. This rules out a screened tail or
short-distance dimensional crossover; a familiar initial hypothesis cannot
cancel the audit. When budget is tight, combine them in one counterfactual
design: use a single factual state plus branches for `p1=0`, doubled `|p2|`,
same-magnitude negative `p1` and `p2`, two distinct nonzero `start_time`
values, and log-spaced radii. Static single-power executors will simply show a
clock-invariant constant slope; time- or scale-dependent worlds will not.
Under observation noise, ordinary unpaired batches do not complete these
audits: keep the factual state tied to a `p1=0` anchor and use common-noise
counterfactual branches so the runtime can reconstruct the `p2` exponent,
both sign relations, all three clock responses, and ordered local radius
slopes.

Three phases are enough to establish clock invariance for a static response.
If they reveal a sign change, however, collect at least four phases and use
the runtime's sinusoidal fit to identify offset, amplitude, angular frequency,
phase, and period. The executable integrator must evaluate that modulation at
absolute time `start_time + t`, not at elapsed `t` alone.

If the runtime reports non-empty `mandatory_audit_blockers`, an early
`<final_law>` will be rejected while rounds and simulator episodes remain.
Continue with a combined matched design until the blocker list is empty.

### 4. Final submission

The final executable is checked deductively against the null-anchored causal
responses already collected in this run. This consumes no new simulator
episode: the runtime fits declared nuisance parameters, executes the proposed
law for a very short interval from audited states, and compares its implied
acceleration with the response graph. If sign, control scaling, radial
normalization, time response, or vector conversion is inconsistent, you may
receive a repair-only turn. In a power law, never use the acceleration measured
at one non-unit radius as the global coefficient; divide out the identified
controls and multiply by `r^q`, then make `fit_parameters` bounds straddle that
normalized coefficient.

In the final round, submit the ordinary `<final_law>` and `<explanation>` plus:

```xml
<scm_final>
{
  "selected_candidate_id": "H_power",
  "remaining_alternatives": ["H_screened"],
  "identified_mechanism": "concise structural mechanism",
  "operator_or_symmetry": "operator/Green function or measured symmetry, with identifiability caveat",
  "source_response_roles": "which controls source the effect and which set response/inertia",
  "scalar_magnitude_law": "|effect| as a function of distance/time/properties",
  "vector_law": "signed vector equation with unit-vector conversion checked",
  "time_and_scale_regimes": "static versus time/history dependent, plus any near/far crossover",
  "remaining_uncertainty": "what the experiments could not identify",
  "evidence_summary": "which interventions and counterexamples decided it",
  "noise_cancelled_scaling_check": "quote and reconcile the controlled p1/p2 exponents, local radius slopes, sign relations, and clock response",
  "operator_correspondence_check": "name the physical operator/Green-function correspondence supported by those invariances, with alternatives and caveats",
  "evidence_consistency_check": "how the final code and explanation preserve the selected candidate's defining features",
  "numerical_robustness_check": "near-collision regularization, integrator success, and replay against collected trajectories"
}
</scm_final>
```

The final law remains the benchmark's executable prediction. The SCM record is
an additional process-level artifact; good prose alone cannot compensate for
poor held-out predictions.

Before submitting, numerically check the final executable law against several
collected trajectories. If it declares continuous free parameters and the
tool is available, complete at least one successful `<run_mse_fit>` before
the final round and carry the fitted values/diagnostics into the submission.
Do not translate a verbal
mechanism into code without checking vector normalization, distance powers,
parameter roles, integration, and the absolute sign from a null-source or
equivalent baseline intervention.

The executable predictor must remain valid through close encounters even when
the structural law is singular. Keep numerical regularization separate from
the claimed physics: use a small positive softening/regularization nuisance
parameter (a broad `1e-3` to `2e-1` fit range is reasonable when the data
contain near encounters), declare it in `fit_parameters()`, and validate it on
the collected crossing trajectories. If using `solve_ivp`, check
`sol.success` and that integration reached the requested duration; never
silently return the last point of an early-terminated solve. Prefer a stable
fixed-step or symplectic fallback when necessary. A law that is verbally
correct but freezes at a singularity is not a valid final prediction.

In the final explanation, state every empirically supported item below:

- the source, response/inertial, hidden, and observation variables;
- the direction/sign of the effect and its dependence on controllable inputs;
- the scalar magnitude law *and* the equivalent vector equation, checking the
  extra factor of distance introduced by a unit vector;
- whether the dynamics are static, time-varying, velocity-dependent, or
  population/latent-class dependent;
- any familiar field operator, Green's function, symmetry, or conservation law
  that is actually distinguished by the evidence.

Do not invent an operator that the experiments cannot distinguish, but do not
omit an operator correspondence that your intervention evidence supports.
In particular, do not collapse a selected screened, fractional/non-local,
multi-regime, time-modulated, retarded, or latent-class candidate into a
generic power law or homogeneous population in the final prose or code.
