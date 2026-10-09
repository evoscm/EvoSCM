### Topology: Anchor + ring orbiters + 5 probes (with masses)

A 2D universe containing 26 particles:
- **Particle 0**: a single "anchor" near the centre with very large inertia — barely responds to forces.
- **Particles 1–20**: 20 background "orbiter" particles arranged on a ring around the anchor. Their masses are drawn from a small fixed set; their initial positions and velocities are fixed by the world.
- **Particles 21–25**: 5 probe particles you fully control — position, velocity, AND mass.

You are scored on how accurately your law predicts the trajectories of the **5 probe particles** (indices 21–25). Probe initial conditions are given to you exactly each round.

### Control Parameters

| Field | Meaning | Typical range |
|---|---|---|
| `probe_positions` | list of 5 `[x, y]` initial coordinates relative to centre | `[-22, 22]²` |
| `probe_velocities` | list of 5 `[vx, vy]` initial velocities | `[-3, 3]²` |
| `probe_masses` (optional) | list of 5 positive floats — each probe's inertial mass; defaults to 1.0 each | use mass-varied probes to test mass dependence |
| `measurement_times` | times at which to record positions and velocities (≤ 10 values) | spans the run |
| `duration` | length of the experiment (interactive only) | ≥ 5.0 |

Avoid placing probes exactly on the anchor (origin) or on top of each other — the singularity produces numerical noise.

### Input Format (interactive mode only)

```
<run_experiment>
[
  {"probe_positions":  [[8,0],[0,8],[-8,0],[0,-8],[10,10]],
   "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]],
   "probe_masses":     [1.0, 1.0, 2.0, 4.0, 1.0],
   "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
]
</run_experiment>
```

You may submit several experiments per round in the JSON array. No comments inside the JSON.

### Output Format

```
<experiment_output>
[
  {
    "measurement_times": [t0, t1, ...],
    "positions":  [[[x0,y0], ..., [x25,y25]], ...],   // shape (T, 26, 2), relative to centre
    "velocities": [[[vx0,vy0], ..., [vx25,vy25]], ...], // shape (T, 26, 2)
    "particle_masses": [m0, m1, ..., m25],             // length-26 mass array
    "background_initial_positions":  [[x,y], ...],     // (21, 2) fixed starting positions
    "background_initial_velocities": [[vx,vy], ...]    // (21, 2) fixed starting velocities
  }
]
```

Particle ordering: index 0 = anchor, indices 1–20 = ring orbiters, indices 21–25 = your probes. Reported masses are clean (no noise).

### `discovered_law` Signature

```python
def discovered_law(positions, velocities, masses, duration):
    # positions:  list of 26 [x, y] — initial positions of every particle (anchor + orbiters + probes)
    # velocities: list of 26 [vx, vy]
    # masses:     list of 26 per-particle masses (these ARE the true masses — no need to discover them)
    # duration:   float — simulate from t = 0 to t = duration
    # return:     list of 26 [x, y] final positions
    # NOTE: scoring is on indices 21-25 (the probes); the others are simulated for context
    return final_positions
```

Your function must simulate **all 26** particles forward (the orbiters and anchor influence the probes through the field), but you are scored only on the final probe positions.
