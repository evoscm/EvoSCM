### Topology: Background population + 5 neutral probes (no masses)

A 2D universe containing a fixed-configuration background population plus 5 neutral probe particles whose initial conditions you control (interactive mode) or that get sampled for you (random mode). The probes feel forces from the field but do **not** generate any field themselves — they are pure measurement instruments.

**Important:** You are scored on how accurately your law predicts the trajectories of the **5 probe particles** (the highest-indexed 5 in the system). The probe initial positions and velocities are given to you exactly each round.


### Control Parameters

| Field | Meaning | Typical range |
|---|---|---|
| `probe_positions` | list of 5 `[x, y]` initial coordinates relative to domain centre | `[-15, 15]²` |
| `probe_velocities` | list of 5 `[vx, vy]` initial velocities | `[-2, 2]²` |
| `measurement_times` | times at which to record positions and velocities (≤ 10 values) | spans the run |
| `duration` | length of the experiment (interactive only) | ≥ 10.0; spread ≥ 10 measurement times across it |

Background-particle initial conditions are fixed by the world; you cannot change them.

### Input Format (interactive mode only)

```
<run_experiment>
[
  {"probe_positions":  [[5,0],[0,5],[-5,0],[0,-5],[7,7]],
   "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]],
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
    "positions":  [[[x0,y0], ..., [xN,yN]], ...],     // shape (T, N_total, 2), relative to centre
    "velocities": [[[vx0,vy0], ..., [vxN,vyN]], ...], // shape (T, N_total, 2)
    "background_initial_positions": [[x,y], ...]      // (N_background, 2) fixed starting positions
  }
]
```

Particle ordering is fixed: low indices = background population, the **last 5** = your probes. Consult the mission description for the exact split.

### `discovered_law` Signature

```python
def discovered_law(positions, velocities, duration):
    # positions:  list of N_total [x, y] — initial positions of every particle (background + probes)
    # velocities: list of N_total [vx, vy]
    # duration:   float — simulate from t = 0 to t = duration
    # return:     list of N_total [x, y] final positions
    # NOTE: scoring is on the LAST 5 entries (the probes); the others are simulated for context
    return final_positions
```

Your function must simulate **all** particles forward (the background dynamics influence the probes through the field), but you are scored only on the final probe positions.
