### Topology: Centre + ring (constrained input)

A 2D universe containing 11 particles:
- **Particle 0**: starts at the centre `[0, 0]`.
- **Particles 1–10**: arranged equally spaced on a ring of radius `ring_radius`, in CCW order.

All 11 particles interact through an unknown field. Unlike most worlds, you do **not** specify per-particle positions or velocities — instead you set two scalars (`ring_radius`, `initial_tangential_velocity`) and the simulator constructs the symmetric initial state from them. Scoring is on all 11 final positions.

### Control Parameters

| Field | Meaning | Typical range |
|---|---|---|
| `ring_radius` | initial distance from centre to ring particles | `[2, 10]` |
| `initial_tangential_velocity` | initial CCW tangential speed for ring particles (centre stays at rest) | `[0, 2]` |
| `measurement_times` | times at which to record positions and velocities (≤ 10 values, in `[0, duration]`) | spans the run |
| `duration` | length of the experiment (interactive only) | ≥ 10.0 |

### Input Format (interactive mode only)

```
<run_experiment>
[
  {"ring_radius": 5.0, "initial_tangential_velocity": 0.0,
   "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
  {"ring_radius": 3.0, "initial_tangential_velocity": 0.0,
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
    "positions":  [[[x0,y0], ..., [x10,y10]], ...],     // shape (T, 11, 2), relative to centre
    "velocities": [[[vx0,vy0], ..., [vx10,vy10]], ...]  // shape (T, 11, 2)
  },
  ...
]
```

Particle ordering is fixed: index 0 = centre, indices 1–10 = ring particles in CCW order. Particle 0 starts at `[0, 0]`.

### `discovered_law` Signature

```python
def discovered_law(positions, velocities, duration, **params):
    # positions:  list of 11 [x, y] — initial positions of all particles (centre + ring)
    # velocities: list of 11 [vx, vy]
    # duration:   float — simulate from t = 0 to t = duration
    # **params:   optional — fitted parameter values injected by the evaluator
    # return:     list of 11 [x, y] final positions
    return final_positions
```

The signature takes raw per-particle initial conditions (not `ring_radius` / `initial_tangential_velocity`) — your law must work from arbitrary positions and velocities, not just the symmetric ring configuration. The `**params` catch-all is optional (only needed if you declare fittable parameters via `fit_parameters()`).
