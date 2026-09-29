import copy
import numpy as np
from scienceagent.release import WORLD_VARIANCES, evaluator_for
from scienceagent.worlds import get_world


def main():
    for world in WORLD_VARIANCES:
        config = get_world(world, engine="nbody", noise_std=0.0, noise_seed=0)
        case = copy.deepcopy(evaluator_for(world, config["executor"]).test_cases[0])
        case["measurement_times"] = [0.0, 0.01]
        result = config["executor"].run([case])

        def check(value):
            if isinstance(value, dict):
                return all(check(x) for x in value.values())
            if isinstance(value, (list, tuple)):
                return all(check(x) for x in value)
            if isinstance(value, (float, int, np.ndarray)):
                return bool(np.isfinite(value).all())
            return True

        assert result and check(result), world
        print(world, "ok", flush=True)


if __name__ == "__main__":
    main()
