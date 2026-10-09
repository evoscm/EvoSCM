### Topology: 2-particle

Two particles in a 2D universe. 

Your task: discover the law of motion governing particle 2.

### Control Parameters

| Field | Meaning | Typical range |
|---|---|---|
| `p1` | scalar property of the source particle | `[0.1, 10]` |
| `p2` | scalar property of the probe particle | `[0.1, 10]` |
| `pos2` | 2D initial position of the probe | `[-10, 10]²` |
| `velocity2` | 2D initial velocity of the probe | `[-5, 5]²` |
| `measurement_times` | times at which to record measurements (≤ 10 values, all in `[0, duration]`) | spans the run |
| `duration` | length of the experiment (interactive only) | ≥ 5.0; use 10.0 to fully resolve the law |

(Some worlds in this topology accept additional optional fields — e.g. a `start_time` if the law of physics varies with absolute time. The mission description will mention any such field; if it is not mentioned, omit it.)

### Input Format (interactive mode only)

```
<run_experiment>
[
  {"p1": ..., "p2": ..., "pos2": [..., ...], "velocity2": [..., ...], "measurement_times": [...]},
  ...
]
</run_experiment>
```

You may submit several experiments per round in the JSON array. No comments inside the JSON.

### Output Format

Each experiment returns:

```
<experiment_output>
[
  {
    "measurement_times": [t0, t1, ...],
    "pos1":      [[x, y], ...],
    "pos2":      [[x, y], ...],
    "velocity1": [[vx, vy], ...],
    "velocity2": [[vx, vy], ...]
  },
  ...
]
```

`pos1` and `velocity1` are reported for completeness even though particle 1 is held fixed.

### `discovered_law` Signature

```python
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    # pos1: [x, y] — always [0, 0] for these worlds
    # pos2: [x, y] — initial position of particle 2
    # p1, p2: scalar properties
    # velocity2: [vx, vy] — initial velocity of particle 2
    # duration: float — simulate from t = 0 to t = duration
    # **params: optional — fitted parameter values injected by the evaluator
    # return: (final_pos2, final_vel2)
    return final_pos2, final_vel2
```

The positional arguments must appear in exactly this order. `**params` is optional (only needed if you declare fittable parameters via `fit_parameters()`).
