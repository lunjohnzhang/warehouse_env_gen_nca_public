import os
import gin
import time
import fire
import torch
import numpy as np
import pandas as pd
import shutil
import multiprocessing
import psutil

# Including this makes gin config work because main imports (pretty much)
# everything.
import env_search.main  # pylint: disable = unused-import

from itertools import repeat

from env_search.analysis.utils import (load_experiment, load_metrics,
                                       load_archive_gen)
from env_search.analysis.visualize_env import visualize_maze, create_movie
from env_search.device import DEVICE
from env_search.utils.logging import setup_logging
from env_search.maze.generator.nca_generator import MazeNCA
from env_search.utils import (format_env_str, read_in_maze_map, flip_tiles,
                              n_params, write_map_str_to_json)
from env_search.maze.level import MazeLevel


def generate_nca_evo_process(all_sols, save_dir):
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)
    all_filenames = [[f"gen_{i:04d}.png"] for i in range(len(all_sols))]
    pool = multiprocessing.Pool(psutil.cpu_count(logical=False))
    pool.starmap(
        visualize_maze,
        zip(
            all_sols,
            all_filenames,
            repeat(save_dir, len(all_sols)),
        ),
    )

    # for i, sol in enumerate(all_sols):
    #     visualize_maze(
    #         sol,
    #         filenames=[f"gen_{i: 04d}.png"],
    #         store_dir=save_dir,
    #     )


def generate_with_time(
    mazeNCA,
    seed_map_int,
    nca_process_dir,
    save=True,
    nca_iter=200,
):
    start_time = time.time()

    out, all_sols = mazeNCA.generate(
        torch.tensor(seed_map_int[np.newaxis, :, :], device=DEVICE),
        n_iter=nca_iter,
        save=save,
    )
    time_elapsed = time.time() - start_time
    generate_nca_evo_process(all_sols, nca_process_dir)
    create_movie(nca_process_dir, "nca_process")

    out = out.squeeze().cpu().numpy()
    print("NCA taken: ", time_elapsed)
    return out, time_elapsed


def offline_run_nca_generator(
    logdir: str,
    seed_env_path: str,
    gen: int = None,
    mode: str = "best",  # one of ["best", "idx"]
    query: "array-like" = None,  # type: ignore
    nca_iter: int = 200,
):
    """
    Load trained NCA and run it once with the specified seed.

    Args:
        logdir: logdir of the experiment
        seed_env_path: path to the NCA seed
        mode:
            1. "best": use best NCA from the archive
            2. "idx" use NCA of specified index by query
        query: specified index
    """
    logdir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    archive = load_archive_gen(logdir, gen)
    df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))

    if mode == "best":
        # Obtain global optima from the archive
        nca_idx = df["objective"].idxmax()
    elif mode == "idx":
        query = tuple(query)
        nca_idx, = np.where(
            df.index_batch() == archive.grid_to_int_index([query])[0])
        if len(nca_idx) == 0:
            raise ValueError(f"Index {query} not available")
        nca_idx = nca_idx[0]
    else:
        raise ValueError(f"Unknown mode {mode}")

    nca_params = np.array(df.filter(regex=("solution_*")).iloc[nca_idx])

    # Read in seed map
    seed_map_str, _ = read_in_maze_map(seed_env_path)
    seed_map_int = MazeLevel.str_to_number(seed_map_str)

    mazeNCA = MazeNCA().to(DEVICE)

    num_params = n_params(mazeNCA)
    print("Number of params: ", num_params)

    env_w, env_h = seed_map_int.shape
    nca_process_dir = logdir.dir(
        f"nca_process_{env_w}x{env_h}_iter={nca_iter}")

    # Set parameter and run
    mazeNCA.set_params(nca_params)
    maze_level, nca_runtime = generate_with_time(
        mazeNCA,
        seed_map_int,
        nca_process_dir,
        nca_iter=nca_iter,
    )

    print("Maze level: ")
    maze_str_grid = MazeLevel(maze_level).to_str_grid()
    for row in maze_str_grid:
        print("\"", end="")
        print("".join(row), end="")
        print("\"")

    maze_str = MazeLevel.number_to_str(maze_level)
    write_map_str_to_json(
        os.path.join(nca_process_dir, "nca_gen.json"),
        maze_str,
        "nca_gen",
        "maze",
        nca_runtime=nca_runtime,
    )


if __name__ == '__main__':
    fire.Fire(offline_run_nca_generator)