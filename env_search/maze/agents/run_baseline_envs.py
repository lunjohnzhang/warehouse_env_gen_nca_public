import os
import fire
import json
import multiprocessing
import numpy as np
import scipy.stats as st

from itertools import repeat
from logdir import LogDir

from env_search.maze.level import MazeLevel
from env_search.utils import read_in_maze_map
from env_search.maze.agents.run_single_rl_agent import run_single_rl_agent


def run_one_baseline(level_filepath, framerate, n_evals, render, logdir):
    curr_result = run_single_rl_agent(
        level_filepath,
        framerate=framerate,
        n_evals=n_evals,
        render=render,
        logdir=logdir,
    )
    maze_name = curr_result["name"]

    with open(logdir.file(f"results/{maze_name}.json"), "w") as f:
        json.dump(curr_result, f, indent=4)
    return curr_result


def run_baseline_envs(
    env_dir: str,
    framerate: int = 24,
    n_evals: int = 100,
    render: bool = False,
    n_workers=2,
):
    logdir = LogDir(f"maze_run_baseline_envs", custom_dir=env_dir)
    envs_dir = logdir.pdir("envs")

    pool = multiprocessing.Pool(n_workers)

    level_filepaths = []

    for filepath in os.listdir(envs_dir):
        level_filepath = os.path.join(envs_dir, filepath)
        level_filepaths.append(level_filepath)

    # Check if the baseline envs overlap
    for i, env1 in enumerate(level_filepaths):
        for j, env2 in enumerate(level_filepaths[i+1:]):
            level1, _ = read_in_maze_map(env1)
            level2, _ = read_in_maze_map(env2)
            level1 = MazeLevel.str_to_number(level1).astype(int)
            level2 = MazeLevel.str_to_number(level2).astype(int)
            same = (level1 == level2).astype(int)
            if np.sum(same) == np.prod(level1.shape):
                print(f"Baseline env {i} and {j} are the same!")

    # Start running
    n_envs = len(level_filepaths)
    results = pool.starmap(
        run_one_baseline,
        zip(
            level_filepaths,
            repeat(framerate, n_envs),
            repeat(n_evals, n_envs),
            repeat(render, n_envs),
            repeat(logdir, n_envs),
        ),
    )

    keys = results[0].keys()
    collected_results = {key: [] for key in keys}
    for result in results:
        for key in keys:
            collected_results[key].append(result[key])

    success_rates = collected_results["success_rate"]
    agg_result = {
        "avg_success_rate": np.mean(success_rates),
        "sem_success_rate": st.sem(success_rates),
    }

    with open(logdir.file(f"agg_result.json"), "w") as f:
        json.dump(agg_result, f, indent=4)


if __name__ == "__main__":
    fire.Fire(run_baseline_envs)