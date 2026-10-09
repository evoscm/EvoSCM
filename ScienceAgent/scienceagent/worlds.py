from scienceagent.executor import (
    SimulationExecutor,
    CircleExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
    NBodySimulationExecutor,
    NBodyCircleExecutor,
    NBodyThreeSpeciesExecutor,
    NBodyDarkMatterExecutor,
    NBodyEtherExecutor,
    NBodyHubbleExecutor,
    NBodyCoulombEasyExecutor,
    NBodyOscillatorExecutor,
    NBodyExtraDimensionsExecutor,
)

_GENERAL_FORM = "$\\dfrac{\\partial^n \\varphi}{\\partial t^n} = L[\\varphi] + S(\\mathrm{particles})$"
_TRUE_LAW_GRAVITY = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = \\nabla^2\\varphi$"
    + "\n$\\mathbf{F} = -\\nabla\\varphi\\,/\\,p_2$"
)
_TRUE_LAW_YUKAWA = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = \\nabla^2\\varphi - \\lambda^{-2}\\varphi,\\quad \\lambda = 2$"
    + "\n$\\mathbf{F} = -\\nabla\\varphi\\,/\\,p_2$"
)
_TRUE_LAW_FRACTIONAL = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = -(-\\nabla^2)^{\\alpha}\\varphi,\\quad \\alpha = 0.75$"
    + "\n$\\mathbf{F} = -\\nabla\\varphi\\,/\\,p_2$"
)
_TRUE_LAW_CIRCLE = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = -(-\\nabla^2)^{\\alpha}\\varphi,\\quad \\alpha = 0.75$"
    + "\n$\\mathbf{F}_i = -\\nabla\\varphi_i$"
    + "\n$\\text{11 particles: 1 center} + \\text{10 ring}$"
)
_TRUE_LAW_THREE_SPECIES = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = \\nabla^2\\varphi$"
    + "\n$\\mathbf{F}_i = -\\nabla\\varphi_i$"
    + "\n$\\text{source\\_coupling}_i = 1.0\\;\\text{(particles 0–9)},\\; 3.0\\;\\text{(particles 10–19)},\\; -2.0\\;\\text{(particles 20–29)}$"
    + "\n$\\text{Probes (30–34): source\\_coupling} = 0$"
)
_TRUE_LAW_DARK_MATTER = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = \\nabla^2\\varphi$"
    + "\n$\\mathbf{F}_i = -\\nabla\\varphi_i$"
    + "\n$\\text{Visible (0–19): source\\_coupling} = 1.0$"
    + "\n$\\text{Dark matter (hidden, 10 particles): source\\_coupling} = 5.0$"
    + "\n$\\text{Probes (20–24 in agent view): source\\_coupling} = 0$"
)
_TRUE_LAW_ETHER = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = \\nabla^2\\varphi$  (sourced only by the central anchor, $Q=50$)"
    + "\n$\\mathbf{F}_i = -\\nabla\\varphi_i + \\alpha\\,m_i\\,\\hat{\\mathbf{y}},\\quad \\alpha = 0.05$"
    + "\n$\\text{20 ring orbiters with masses}\\in\\{1,2,4\\};\\;\\text{5 probes (test particles)}$"
)
_TRUE_LAW_HUBBLE = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = \\nabla^2\\varphi$  (sourced only by the central anchor, $Q=50$)"
    + "\n$\\mathbf{a}_i = -\\nabla\\varphi_i / m_i + H\\,\\mathbf{r}_i,\\quad H = 0.05$"
    + "\n$r_{\\rm crit} = \\sqrt{Q/(2\\pi H)} \\approx 12.6$"
    + "\n$\\text{20 ring orbiters with masses}\\in\\{1,2,4\\};\\;\\text{5 probes (test particles)}$"
)
_TRUE_LAW_COULOMB_EASY = (
    "$\\mathbf{F}_{12} = -k\\,q_1 q_2\\,\\hat{\\mathbf{r}}_{12} / r^2,\\quad k = 1$"
    + "\n$q_0 = +|p_1|,\\; q_1 = -|p_2|$ (signs hidden, always attractive)"
    + "\n$\\text{Particle 0 pinned at origin; particle 1 mobile, inertia 1}$"
)
_TRUE_LAW_OSCILLATOR = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$L[\\varphi] = G(t)\\,\\nabla^2\\varphi,\\quad G(t) = G_0\\cos(\\omega t + \\phi)$"
    + "\n$G_0 = 5,\\;\\omega = \\pi/2\\;(T = 4),\\;\\phi = 0$"
    + "\n$\\mathbf{F}_2 = -\\nabla\\varphi\\,/\\,p_2$"
)
_TRUE_LAW_EXTRA_DIMENSIONS = (
    _GENERAL_FORM
    + "\n\n$n = 0$"
    + "\n$\\mathbf{F}_2(r) = \\dfrac{G\\,L\\,p_1}{4\\pi}\\sum_{n\\in\\mathbb{Z}}\\dfrac{r}{(r^2+(nL)^2)^{3/2}}\\,(-\\hat{\\mathbf{r}}),\\quad L = 2\\pi R_c$"
    + "\n$G = 1,\\;R_c = 0.5$"
    + "\n$r \\gg R_c:\\; F \\to p_1/(2\\pi r)$ (2D Poisson — looks like gravity)"
    + "\n$r \\ll R_c:\\; F \\to G L p_1/(4\\pi r^2) = R_c p_1/(2 r^2)$ (3D Newton)"
)
_RUBRIC_GRAVITY = "10 — Identifies a static scalar field with a Laplacian/Poisson operator\n     (∇²φ = source); states the resulting attractive force falls off as 1/r\n     (not 1/r²) in 2D, or equivalently that the Green's function is\n     logarithmic; correctly distinguishes p1 as source coupling and p2 as\n     particle inertia.\n 7–9 — Identifies a static attractive field with roughly correct decay, but\n     asserts the 3D-Newtonian 1/r² falloff, muddles the p1/p2 roles, or\n     describes the operator only qualitatively without naming it.\n 4–6 — Recognises static attraction but commits to a clearly wrong decay law\n     (exponential, inverse-cube), or fails to distinguish source coupling\n     from inertia.\n 1–3 — Wrong temporal character (time-evolving or wave-like) or a\n     qualitatively wrong force law (repulsive, constant).\n   0 — Empty, irrelevant, or no physical content."
_RUBRIC_YUKAWA = "10 — Identifies a static screened / Helmholtz operator\n     (∇²φ − φ/λ² = source, or equivalent Yukawa form); names or describes\n     exponential suppression at long range with screening length within\n     roughly 1–4 (ground truth λ = 2); correctly identifies p1 as source\n     coupling and p2 as inertia.\n 7–9 — Identifies screened / Yukawa-like behaviour qualitatively (short-range\n     attraction, long-range suppression) but misses a quantitative detail\n     (λ badly off or not estimated) or muddles the p1/p2 roles.\n 4–6 — Recognises a static attractive field but treats it as a plain\n     Laplacian or generic power-law and misses the screening structure, or\n     asserts screening without any characterisation of its scale.\n 1–3 — Wrong operator family (time-evolving, wave) or no distance-dependent\n     suppression despite the clear experimental signature.\n   0 — Empty or irrelevant."
_RUBRIC_FRACTIONAL = "10 — Identifies a static non-local / fractional-Laplacian operator of the\n     form −(−∇²)^α with α in a plausible range (roughly 0.3–0.8; ground\n     truth α = 0.5); describes the force as decaying more slowly with\n     distance than the standard 2D Laplacian case (enhanced long-range\n     interaction); correctly names p1 as source and p2 as inertia.\n 7–9 — Identifies an anomalous / non-local spatial operator with unusual\n     (slower-than-standard) long-range behaviour, but misses α, places α\n     outside a plausible range, or articulates the direction of the anomaly\n     vaguely or incorrectly.\n 4–6 — Recognises static attraction but assumes a standard Laplacian or\n     ordinary power-law, missing the non-local / fractional character.\n 1–3 — Wrong operator family (time-evolving or repulsive) or no\n     characterisation of the spatial operator at all.\n   0 — Empty or irrelevant."
_RUBRIC_EXTRA_DIMENSIONS = "10 — Identifies that the visible 2D world has *one* extra spatial dimension\n     compactified at radius R; states the force is the Kaluza-Klein image\n     sum over the compact dimension, ∝ Σ_n r/(r²+(nL)²)^(3/2) with\n     L = 2π R; correctly identifies the two asymptotic regimes\n     (r ≫ R: 2D Poisson F ∝ p1/(2π r); r ≲ R: 3D Newtonian\n     F ∝ p1/(4π r²) becoming exponentially divergent at r → 0); estimates\n     R within roughly a factor of 2 of 0.5 (i.e. R ∈ [0.2, 1.5]) and gets\n     the long-range coupling within ~30 % of G = 1; correctly assigns p1\n     as source coupling and p2 as inertia.\n 7–9 — Recognises a *crossover* in the force law from 1/r at long range to\n     a steeper power (≈ 1/r²) at short range, and offers a quantitative\n     compactification scale within roughly an order of magnitude of 0.5,\n     but misses the geometric image-sum structure or doesn't connect the\n     short-range regime to a higher-dimensional Newtonian law.\n 4–6 — Notices anomalous deviation from a pure 1/r force at short range\n     (or anomalous deviation from pure 1/r² at long range) but proposes a\n     wrong functional form (e.g. Yukawa screening, fractional Laplacian,\n     pure power law of intermediate exponent) without identifying a true\n     dimensional crossover.\n 1–3 — Fits the data with a single canonical law (plain 2D gravity, plain\n     3D gravity, fractional, Yukawa, …) without acknowledging that the\n     same law fails at the opposite end of the distance range.\n   0 — Empty or irrelevant."
_RUBRIC_OSCILLATOR = "10 — Identifies the standard 2D Laplacian spatial form (∇²φ = source, force\n     ∝ 1/r in 2D) but with a *time-modulated overall coupling* G(t) — i.e.\n     the strength (and SIGN) of the interaction depends on absolute time,\n     not just on geometry; explicitly notes that the same configuration is\n     attractive at some phases and *repulsive* at others; identifies a\n     sinusoidal modulation G(t) = G₀·cos(ω t + φ) (or sin equivalent) and\n     estimates the period within roughly a factor of 2 of T = 4; correctly\n     keeps p1 as source coupling and p2 as inertia.  Note: the amplitude\n     can be reported either as G₀ ≈ 5 with an explicit 1/(2π) Green's\n     function prefactor, or as a single combined coefficient\n     G₀/(2π) ≈ 0.80 absorbing the Poisson kernel into the amplitude —\n     these are the same physics expressed in different conventions and\n     should be scored equivalently as long as the effective coupling is\n     numerically correct.\n 7–9 — Identifies time-varying coupling on top of a 2D-gravity-like spatial\n     form, but misses the period quantitatively, omits the sign-flipping\n     behaviour, or muddles the p1/p2 roles.\n 4–6 — Recognises that the dynamics are time-dependent but treats the\n     variation as transient / decaying or otherwise non-periodic, OR\n     identifies the 1/r spatial form but treats the coupling as constant\n     (missing the entire time modulation).\n 1–3 — Models the system as static gravity, or proposes a qualitatively\n     wrong mechanism (drag, noise, history-dependent diffusion, …) that\n     does not include an explicit dependence on absolute time.\n   0 — Empty or irrelevant."
_RUBRIC_CIRCLE = "10 — Identifies a static non-local / fractional-Laplacian operator\n     −(−∇²)^α with α in a plausible range (roughly 0.5–1.0; ground truth\n     α = 0.75); states that the coupling is uniform across all 11 particles;\n     describes force behaviour intermediate between the logarithmic 2D\n     Laplacian case and pure long-range attraction.\n 7–9 — Identifies a non-local / anomalous-decay operator, but estimates α\n     outside a plausible range, omits the uniform-coupling claim, or\n     muddles the direction of the anomaly.\n 4–6 — Recognises attractive static interactions but assumes a standard\n     Laplacian (or Newtonian 1/r) and misses the fractional character.\n 1–3 — Wrong operator family (time-evolving, repulsive) or no spatial\n     operator named.\n   0 — Empty or irrelevant."
_RUBRIC_THREE_SPECIES = "10 — Identifies a static Laplacian with force −∇φ; identifies three distinct\n     source species among particles 0–29 plus 5 neutral probes (30–34);\n     correctly identifies one species as repulsive (negative coupling);\n     estimates coupling ratios approximately matching +1 : +3 : −2 (each\n     within roughly a factor of 2, with signs correct); identifies the\n     probes as having approximately zero source coupling.\n 7–9 — Identifies the Laplacian and the three-species + probe structure\n     with correct signs, but coupling magnitudes are off, probes are lumped\n     with one of the species, or the particle-index partitioning is\n     slightly wrong.\n 4–6 — Identifies a Laplacian field and some species structure, but misses\n     the repulsive species (all three treated as attractive), finds only\n     two species, or fails to distinguish the neutral probes.\n 1–3 — Treats all particles as identical, or posits a wrong operator family.\n   0 — Empty or irrelevant."
_RUBRIC_ETHER = "10 — Identifies a static 2D Laplacian central force sourced by the single\n     anchor particle (index 0); identifies the 20 orbiters and 5 probes as\n     test particles responding to that field; identifies a uniform northward\n     drift acceleration on every particle (or, equivalently, a body-force\n     proportional to mass producing a mass-independent acceleration);\n     estimates the drift acceleration α within roughly a factor of 2 of\n     0.05; recognises that orbiter masses ∈ {1, 2, 4} but that with the\n     mass-proportional ether force the drift looks identical for all\n     particles in absolute coordinates.\n 7–9 — Identifies the central Laplacian + uniform northward drift, but\n     misses the F ∝ m / a = const equivalence, gets α badly off, or fails\n     to identify which particle is the anchor.\n 4–6 — Identifies the central attraction OR the drift but not both, or\n     mis-attributes the drift to a directional Laplacian / wind / repulsion\n     between specific particles.\n 1–3 — Wrong operator family, no drift identified, or no central anchor\n     identified despite the obvious common parabolic envelope.\n   0 — Empty or irrelevant."
_RUBRIC_HUBBLE = '10 — Identifies a static 2D Laplacian central force sourced by the single\n     anchor particle (index 0); identifies the 20 orbiters and 5 probes as\n     test particles; identifies a *position-dependent* outward body-force\n     that grows linearly with distance from the anchor (a = H · r,\n     mass-independent), recognises a critical radius beyond which probes\n     accelerate outward and orbits unbind, and estimates H within roughly\n     a factor of 2 of 0.05.\n 7–9 — Identifies the central Laplacian and an outward repulsive effect at\n     large radii, but misses the linear-in-r structure (e.g. assumes a\n     constant outward force), gets H badly off, or attributes the outward\n     push to a particular particle rather than to space itself.\n 4–6 — Identifies the central attraction but treats the outward effect\n     qualitatively only — e.g. notes "orbits unbind at large r" without\n     distinguishing a body-force from a missing-mass / dark-matter\n     hypothesis.\n 1–3 — Wrong operator family, or interprets the outward push as random\n     noise, drag, or some pairwise repulsion between probes/orbiters\n     despite the obvious radial-from-anchor pattern.\n   0 — Empty or irrelevant.'
_RUBRIC_COULOMB_EASY = "10 — Identifies a static central pairwise attractive force with 1/r²\n     (inverse-square) falloff; correctly identifies the role of p1 and p2\n     as charges (or charge magnitudes) entering the force as a product\n     (F ∝ p1 · p2 / r²); states that particle 1 is held fixed and that\n     particle 2's inertia is 1; estimates the coupling/strength constant\n     within roughly a factor of 2 of 1.\n 7–9 — Identifies an attractive 1/r² central force, but is vague or wrong\n     about how p1 and p2 enter (e.g. claims one of them is inertia, or\n     that the force depends additively on them), or omits the strength\n     estimate.\n 4–6 — Recognises static attraction but identifies a clearly wrong falloff\n     (1/r, exponential, constant), or proposes a 1/r² law without\n     identifying p1 and p2 as charges.\n 1–3 — Wrong qualitative behaviour (repulsive, time-evolving, wave-like)\n     or no mention of distance scaling.\n   0 — Empty, irrelevant, or no physical content."
_RUBRIC_DARK_MATTER = "10 — Identifies a static Laplacian with force −∇φ; concludes that hidden /\n     unseen sources exist based on visible particles accelerating toward\n     apparently empty regions; estimates the hidden population roughly\n     correctly (count in 5–15; ground truth 10) with coupling stronger than\n     the visible population (roughly 3–8×; ground truth 5×); identifies the\n     probes (agent indices 20–24) as neutral (non-sourcing but responsive).\n 7–9 — Identifies the Laplacian and the existence of hidden sources, but\n     gets their count or coupling strength badly wrong, or fails to\n     characterise the probes as neutral.\n 4–6 — Identifies a Laplacian field but attributes the visible particles'\n     anomalous behaviour to noise, the probes, or measurement error rather\n     than to hidden sources.\n 1–3 — Wrong operator family, or a fundamentally wrong mechanism (e.g.\n     dynamical instability with no hidden matter).\n   0 — Empty or irrelevant."
WORLDS = {
    "gravity": {
        "description": "Classic inverse-square-law-like attraction mediated by a 2D Laplacian field.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_kwargs": {
            "operators": [{"type": "laplacian", "params": {"strength": 1.0}}],
            "temporal_order": 0,
        },
        "true_law": _TRUE_LAW_GRAVITY,
        "true_law_title": "True Laplacian",
        "optimal_explanation": "Two particles interact through a static scalar field obeying the 2D Poisson equation, ∇²φ = source, where particle 1 sources the field with strength p1 and particle 2 is accelerated by -∇φ divided by its inertia p2. Because the Laplacian Green's function is logarithmic in 2D, the resulting attractive force falls off as 1/r rather than 1/r².",
        "explanation_rubric": _RUBRIC_GRAVITY,
    },
    "yukawa": {
        "description": "Screened (Yukawa) potential — exponentially suppressed at long range.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_kwargs": {
            "operators": [
                {
                    "type": "screening",
                    "params": {"strength": 1.0, "screening_length": 2.0},
                }
            ],
            "temporal_order": 0,
        },
        "true_law": _TRUE_LAW_YUKAWA,
        "true_law_title": "True Yukawa",
        "optimal_explanation": "Two particles interact through a screened scalar field obeying a 2D Helmholtz equation, ∇²φ - φ/λ² = source, with screening length λ = 2. Particle 1 sources the field with strength p1 and particle 2 accelerates under -∇φ/p2. The force is attractive at short range but suppressed exponentially beyond ~2 length units.",
        "explanation_rubric": _RUBRIC_YUKAWA,
    },
    "fractional": {
        "description": "Fractional Laplacian — anomalous power-law force.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_kwargs": {
            "operators": [
                {
                    "type": "fractional_laplacian",
                    "params": {"strength": 1.0, "alpha": 0.5},
                }
            ],
            "temporal_order": 0,
        },
        "true_law": _TRUE_LAW_FRACTIONAL,
        "true_law_title": "True Fractional Laplacian",
        "optimal_explanation": "Two particles interact through a static field governed by a fractional Laplacian operator, -(-∇²)^α with α = 0.5, sourced by particle 1 with coupling p1. The non-local operator produces a force on particle 2 that decays more slowly with distance than the standard 2D Laplacian case — long-range interactions are enhanced. Particle 2 responds with inertia p2 and the field is time-independent.",
        "explanation_rubric": _RUBRIC_FRACTIONAL,
    },
    "oscillator": {
        "description": "2D Poisson force (1/r in 2D) whose overall coupling oscillates sinusoidally with absolute time — the law of physics itself varies, with period T = 4 and a coupling that periodically changes sign.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "OscillatorExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_OSCILLATOR,
        "true_law_title": "True Time-Modulated Laplacian",
        "optimal_explanation": "Two particles interact through a 2D Poisson field, ∇²φ = source, with particle 1 the source (coupling p1) and particle 2 a test particle (inertia p2), so the spatial part of the force is the standard 2D 1/r law, F = -∇φ/p2.  The overall coupling of this field is modulated in absolute time by a sinusoid G(t) = G₀·cos(ω t + φ) with G₀ = 5, ω = π/2 (period T = 4), and φ = 0, so the effective force is G(t) times the static 2D-Poisson force.  Because G(t) changes sign within each period, the same particle pair attracts for a quarter-period, exerts no force at the zero-crossings, and repels for the next quarter — identical configurations evolve into qualitatively different trajectories depending on the absolute time at which the experiment is performed.",
        "explanation_rubric": _RUBRIC_OSCILLATOR,
        "experiment_format": '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0, 2.0], "start_time": 0.0}]</run_experiment>',
    },
    "extra_dimensions": {
        "description": "Extra-dimension force law: 2D-Poisson 1/r at long range, 3D Newtonian 1/r² below the compactification scale R_c = 0.5.  Identical to the 'gravity' world for any experiment confined to r ≳ a few R_c.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "ExtraDimensionsExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_EXTRA_DIMENSIONS,
        "true_law_title": "True Kaluza-Klein extra-dimension force",
        "optimal_explanation": "Two particles interact through a static force law that comes from a 2D visible universe with one extra spatial dimension compactified on a circle of radius R_c = 0.5 (so the compact circumference is L = 2π R_c ≈ 3.14).  Particle 1 sits fixed at the origin with source coupling p1; particle 2 has inertia p2 and feels the pairwise force\n\n    F(r) = (G·L)/(4π) · p1 · Σ_n r / (r² + (nL)²)^(3/2)\n\nwhere the sum runs over the infinite tower of image charges of particle 1 along the compact dimension and G = 1.  Two asymptotic regimes:\n  • r ≫ R_c: the sum becomes an integral and the force reduces to F → p1 / (2π r) — the standard 2D Poisson 1/r law indistinguishable from the gravity world.\n  • r ≲ R_c: only the n = 0 image survives and the force becomes F → G·L·p1 / (4π r²) = R_c·p1 / (2 r²) — a 3D Newtonian inverse-square law.\nAn agent that only probes typical separations (r ≈ 3–5) sees essentially pure 2D gravity; the extra dimension only reveals itself in experiments with r ≲ 1, where the force grows as 1/r² rather than 1/r.",
        "explanation_rubric": _RUBRIC_EXTRA_DIMENSIONS,
    },
    "dark_matter": {
        "description": "Standard Laplacian with 20 visible particles, 10 invisible dark matter (source=5), and 5 probes.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "DarkMatterExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_DARK_MATTER,
        "true_law_title": "True Laplacian (dark matter)",
        "optimal_explanation": "The system obeys a static 2D Laplacian field, ∇²φ = source, with force -∇φ on each particle, but contains hidden structure the agent cannot directly observe: 10 dark-matter particles with source coupling 5.0 — five times stronger than the visible population — whose positions are concealed. The 20 visible particles all have source coupling 1.0 and the 5 probes are neutral (coupling 0). Visible particles appear to accelerate toward empty regions because those regions actually contain unseen dark-matter sources, and probes respond to the combined visible+dark field.",
        "explanation_rubric": _RUBRIC_DARK_MATTER,
        "law_stub": "def discovered_law(positions, velocities, duration):\n    # positions: list of 25 [x, y] coords relative to center\n    #   indices 0-19: visible background, 20-24: probes\n    # velocities: list of 25 [vx, vy]\n    # duration: float, simulate from t=0 to t=duration\n    # return: list of 25 [x, y] final positions\n    # NOTE: you are scored on the 5 PROBE trajectories (indices 20-24)\n    return final_positions\n",
        "experiment_format": '<run_experiment>[{"probe_positions": [[5,0],[0,5],[-5,0],[0,-5],[7,7]], "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]</run_experiment>',
    },
    "three_species": {
        "description": "Standard Laplacian with 30 particles of 3 hidden species (source couplings 1, 3, -2) + 5 neutral probes.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "ThreeSpeciesExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_THREE_SPECIES,
        "true_law_title": "True Laplacian (three species)",
        "optimal_explanation": "Thirty-five particles interact through a static Laplacian field, ∇²φ = source, with acceleration -∇φ. The 30 background particles split into three hidden species: particles 0–9 with source coupling +1, particles 10–19 with +3 (strong attractors), and particles 20–29 with -2 (repulsive — they source a field that pushes other particles away rather than pulling them in). Particles 30–34 are neutral probes with zero coupling, feeling forces without sourcing the field.",
        "explanation_rubric": _RUBRIC_THREE_SPECIES,
        "law_stub": "def discovered_law(positions, velocities, duration):\n    # positions: list of 35 [x, y] coords relative to center\n    # velocities: list of 35 [vx, vy]\n    # duration: float, simulate from t=0 to t=duration\n    # return: list of 35 [x, y] final positions\n    return final_positions\n",
        "experiment_format": '<run_experiment>[{"probe_positions": [[5,0],[0,5],[-5,0],[0,-5],[7,7]], "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]</run_experiment>',
    },
    "ether": {
        "description": "2D Laplacian central anchor + 20 orbiters with masses {1,2,4} + 5 probes, with a uniform northward 'ether' drift acceleration on every particle.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "EtherExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_ETHER,
        "true_law_title": "True Laplacian + Ether drift",
        "optimal_explanation": "Twenty-six particles interact through a static 2D Laplacian field, ∇²φ = source, sourced only by the central anchor (index 0) with coupling 50. The 20 orbiters (masses cycled through 1, 2, 4) and 5 probes are test particles (zero source coupling) feeling -∇φ from the anchor. Layered on top is a uniform 'ether' field that exerts a body-force F = α·m·ŷ on every particle, with α ≈ 0.05; because the force is exactly proportional to mass, every particle picks up the same northward acceleration α regardless of mass — a parabolic drift common to anchor, orbiters, and probes alike, on top of the orbital motion.",
        "explanation_rubric": _RUBRIC_ETHER,
        "law_stub": "def discovered_law(positions, velocities, masses, duration):\n    # positions: list of 26 [x, y] coords relative to centre\n    # velocities: list of 26 [vx, vy]\n    # masses: list of 26 per-particle masses\n    # duration: float, simulate from t=0 to t=duration\n    # return: list of 26 [x, y] final positions\n    # NOTE: scoring focuses on the 5 PROBE trajectories (indices 21-25)\n    return final_positions\n",
        "experiment_format": '<run_experiment>[{"probe_positions": [[8,0],[0,8],[-8,0],[0,-8],[10,10]], "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], "probe_masses": [1.0, 1.0, 2.0, 4.0, 1.0], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]</run_experiment>',
    },
    "hubble": {
        "description": "2D Laplacian central anchor + 20 orbiters with masses {1,2,4} + 5 probes, with a position-dependent radial 'Hubble flow' that pushes every particle outward proportional to its distance from the centre.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "HubbleExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_HUBBLE,
        "true_law_title": "True Laplacian + Hubble flow",
        "optimal_explanation": "Twenty-six particles interact through a static 2D Laplacian field, ∇²φ = source, sourced only by the central anchor (index 0) with coupling 50. The 20 orbiters (masses cycled through 1, 2, 4) and 5 probes are test particles (zero source coupling) feeling -∇φ from the anchor. Layered on top is a Hubble-flow body-force that gives every particle an additional outward radial acceleration a = H · r with H ≈ 0.05, where r is the displacement from the centre. Because the force is mass-independent, the same H acts on every particle. The critical radius where Hubble outward push balances central inward gravity is r_crit = √(Q/(2πH)) ≈ 12.6: probes at smaller r remain bound and orbit (with slightly reduced effective gravity), while probes outside r_crit accelerate outward and escape.",
        "explanation_rubric": _RUBRIC_HUBBLE,
        "law_stub": "def discovered_law(positions, velocities, masses, duration):\n    # positions: list of 26 [x, y] coords relative to centre\n    # velocities: list of 26 [vx, vy]\n    # masses: list of 26 per-particle masses\n    # duration: float, simulate from t=0 to t=duration\n    # return: list of 26 [x, y] final positions\n    # NOTE: scoring focuses on the 5 PROBE trajectories (indices 21-25)\n    return final_positions\n",
        "experiment_format": '<run_experiment>[{"probe_positions": [[5,0],[10,0],[15,0],[18,0],[0,12]], "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], "probe_masses": [1.0, 1.0, 2.0, 4.0, 1.0], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]</run_experiment>',
    },
    "circle": {
        "description": "Fractional Laplacian (alpha=0.75) — 11 particles, 1 center + 10 ring.",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "CircleExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_CIRCLE,
        "true_law_title": "True Fractional Laplacian (circle)",
        "optimal_explanation": "Eleven particles — one at the center plus ten arranged on a surrounding ring — interact through a static field governed by a fractional Laplacian operator -(-∇²)^α with α = 0.75. The force on each particle is -∇φ, where φ is sourced by all particles with uniform coupling. The non-local fractional operator produces a force-versus-distance law that is intermediate between the logarithmic 2D Laplacian and pure long-range behavior.",
        "explanation_rubric": _RUBRIC_CIRCLE,
        "law_stub": "def discovered_law(positions, velocities, duration):\n    # positions: list of 11 [x, y] coords relative to center\n    # velocities: list of 11 [vx, vy]\n    # duration: float, simulate from t=0 to t=duration\n    # return: list of 11 [x, y] final positions\n    return final_positions\n",
        "experiment_format": '<run_experiment>[{"ring_radius": 5.0, "initial_tangential_velocity": 0.0, "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]</run_experiment>',
    },
    "coulomb_easy": {
        "description": "2-particle attractive Coulomb world (1/r², N-body only).",
        "mission": "You are an expert physics and AI research scientist tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyse the data they return, and ultimately deduce the underlying scientific law. Note that the laws of physics in this universe may differ from those in our own. You can perform experiments to gather data but must follow the protocol strictly.",
        "executor_class": "CoulombEasyExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_COULOMB_EASY,
        "true_law_title": "True Coulomb (attractive)",
        "optimal_explanation": "Two particles interact through a central, attractive 1/r² force, F = k · p1 · p2 / r². Particle 1 is pinned at the origin and particle 2 (inertia 1) is accelerated along the line of separation with magnitude proportional to the product of the two charges. Underneath, the world uses standard Coulomb's law with hidden opposite signs (q_0 = +|p1|, q_1 = -|p2|), so attraction is guaranteed regardless of the input signs.",
        "explanation_rubric": _RUBRIC_COULOMB_EASY,
        "law_stub": "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n    # pos1: [x, y] position of fixed particle 1 (always [0, 0])\n    # pos2: [x, y] initial position of particle 2\n    # p1, p2: scalar charges\n    # velocity2: [vx, vy] initial velocity of particle 2\n    # duration: float, simulate from t=0 to t=duration\n    # return: (final_pos2, final_vel2)\n    return final_pos2, final_vel2\n",
        "experiment_format": '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.5], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0]}]</run_experiment>',
    },
}
_NBODY_EXECUTOR_CLASSES = {
    "SimulationExecutor": NBodySimulationExecutor,
    "CircleExecutor": NBodyCircleExecutor,
    "ThreeSpeciesExecutor": NBodyThreeSpeciesExecutor,
    "DarkMatterExecutor": NBodyDarkMatterExecutor,
    "EtherExecutor": NBodyEtherExecutor,
    "HubbleExecutor": NBodyHubbleExecutor,
    "CoulombEasyExecutor": NBodyCoulombEasyExecutor,
    "OscillatorExecutor": NBodyOscillatorExecutor,
    "ExtraDimensionsExecutor": NBodyExtraDimensionsExecutor,
}
_FIELD_EXECUTOR_CLASSES = {
    "SimulationExecutor": SimulationExecutor,
    "CircleExecutor": CircleExecutor,
    "ThreeSpeciesExecutor": ThreeSpeciesExecutor,
    "DarkMatterExecutor": DarkMatterExecutor,
}
_INSTRUCTIONS_BY_EXECUTOR = {
    "SimulationExecutor": "PhysicsSchool/prompts/2particle_instructions.md",
    "OscillatorExecutor": "PhysicsSchool/prompts/2particle_instructions.md",
    "ExtraDimensionsExecutor": "PhysicsSchool/prompts/2particle_instructions.md",
    "CoulombEasyExecutor": "PhysicsSchool/prompts/2particle_instructions.md",
    "DarkMatterExecutor": "PhysicsSchool/prompts/probes_instructions.md",
    "ThreeSpeciesExecutor": "PhysicsSchool/prompts/probes_instructions.md",
    "EtherExecutor": "PhysicsSchool/prompts/probes_with_masses_instructions.md",
    "HubbleExecutor": "PhysicsSchool/prompts/probes_with_masses_instructions.md",
    "CircleExecutor": "PhysicsSchool/prompts/circle_instructions.md",
}


def get_world(name: str, engine: str = "field", **executor_overrides) -> dict:
    if name not in WORLDS:
        raise ValueError(f"Unknown world '{name}'. Available: {list(WORLDS)}")
    if engine not in ("field", "nbody"):
        raise ValueError(f"engine must be 'field' or 'nbody', got {engine!r}")
    entry = WORLDS[name]
    kwargs = {**entry["executor_kwargs"], **executor_overrides}
    executor_class_name = entry.get("executor_class", "SimulationExecutor")
    if engine == "nbody":
        if executor_class_name not in _NBODY_EXECUTOR_CLASSES:
            raise ValueError(
                f"World {name!r} has no NBody twin (engine='nbody' supports only worlds with temporal_order=0 and a single static PDE operator).  Use engine='field' instead."
            )
        if kwargs.get("temporal_order", 0) != 0:
            raise ValueError(
                f"engine='nbody' cannot run world {name!r}: it requires a time-evolving field (temporal_order != 0)."
            )
        executor_cls = _NBODY_EXECUTOR_CLASSES[executor_class_name]
    else:
        if executor_class_name not in _FIELD_EXECUTOR_CLASSES:
            raise ValueError(
                f"World {name!r} has no FieldSampler implementation (its physics — e.g. uniform body-forces — has no equivalent in the FFT operator set). Use engine='nbody' instead."
            )
        executor_cls = _FIELD_EXECUTOR_CLASSES[executor_class_name]
    executor = executor_cls(**kwargs)
    default_law_stub = "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n    # your best implementation\n    return final_pos2, final_vel2\n"
    default_system_prompt = "PhysicsSchool/prompts/_template_interactive.md"
    default_experiment_format = '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0, 2.0]}]</run_experiment>'
    instructions_path = entry.get(
        "instructions", _INSTRUCTIONS_BY_EXECUTOR.get(executor_class_name)
    )
    return {
        "executor": executor,
        "mission": entry["mission"],
        "true_law": entry["true_law"],
        "true_law_title": entry["true_law_title"],
        "optimal_explanation": entry.get("optimal_explanation", ""),
        "explanation_rubric": entry.get("explanation_rubric", ""),
        "system_prompt": entry.get("system_prompt", default_system_prompt),
        "instructions": instructions_path,
        "law_stub": entry.get("law_stub", default_law_stub),
        "experiment_format": entry.get("experiment_format", default_experiment_format),
    }
