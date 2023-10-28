import fire
import numpy as np
import multiprocessing

from itertools import repeat
from logdir import LogDir
from scipy.sparse import csgraph
from env_search.utils import (read_in_maze_map, maze_env_str2number,
                              maze_env_number2str, maze_obj_types,
                              write_map_str_to_json)
from env_search.maze.module import MazeModule


def get_max_path_len(maze_np):
    # Path length calculation
    adj = MazeModule._get_adj(maze_np)

    # Find the best distances
    dist, predecessors = csgraph.floyd_warshall(adj, return_predecessors=True)
    dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter
    return dist.max()


# def generate_one_env(n_tiles, n_wall, comp_maze_np):

#     curr_path_len = get_max_path_len(curr_env)
#     return curr_env, curr_path_len


def random_generate(
    compare_maze_filepath,
    n_gen=10,
    path_len_low_tol=0.5,
    path_len_up_tol=1.5,
    n_workers=32,
    batch_size=100,
):
    """
    Generate `n_gen` maze environments that have the same number of walls and
    longest path length within the `path_len_tol` as the maze environment in
    `compare_maze_filepath`
    """
    # Read in compare maze
    comp_maze_str, comp_name = read_in_maze_map(compare_maze_filepath)
    comp_maze_np = maze_env_str2number(comp_maze_str)
    tile_ele, tile_cnt = np.unique(comp_maze_np, return_counts=True)
    tile_cnt_dict = dict(zip(tile_ele, tile_cnt))
    n_wall = tile_cnt_dict[maze_obj_types.index("X")]
    comp_path_len = get_max_path_len(comp_maze_np)
    n_tiles = np.prod(comp_maze_np.shape)

    print(f"Comp env: n_wall={n_wall}, max_path_len={comp_path_len}")

    # Create logdir
    logdir = LogDir(f"baseline_maze_map_{comp_name}", rootdir="./logs")

    pool = multiprocessing.Pool(n_workers)

    have_gen = 0
    tried_batch = 0
    maze_envs = []
    while have_gen < n_gen:
        batch_envs = []
        for i in range(batch_size):
            curr_env = np.zeros(n_tiles, dtype=int)
            curr_env[:] = maze_obj_types.index(" ")

            # Sample `n_wall` places to put the wall
            choices = np.arange(n_tiles)
            sampled_idx = np.random.choice(choices, n_wall, replace=False)
            curr_env[sampled_idx] = maze_obj_types.index("X")
            curr_env = curr_env.reshape(comp_maze_np.shape)
            batch_envs.append(curr_env)

        all_max_path_len = pool.starmap(
            get_max_path_len,
            zip(batch_envs),
        )
        tried_batch += 1
        all_path_lens = []

        for curr_env, curr_path_len in zip(batch_envs, all_max_path_len):
            all_path_lens.append(curr_path_len)
            if path_len_low_tol * comp_path_len <= curr_path_len <= path_len_up_tol * comp_path_len:
                maze_envs.append(curr_env)
                have_gen += 1
                if have_gen >= n_gen:
                    break
        # all_path_lens = np.array(all_path_lens)
        print(f"After batch {tried_batch}, got {have_gen} envs")
        print(f"Avg length of curr batch: {np.mean(all_path_lens)}")
        print(f"Min length of curr batch: {np.min(all_path_lens)}")
        print(f"Max length of curr batch: {np.max(all_path_lens)}")


    # Save generated envs in logdir
    for i, maze_env in enumerate(maze_envs):
        map_name = f"baseline_{i}"
        write_map_str_to_json(
            logdir.pfile(f"envs/{map_name}.json"),
            maze_env_number2str(maze_env),
            map_name,
            "maze",
        )

    print(
        f"Generated {n_gen} envs of specified difficulty. Tried {tried_batch} batches with batch size {batch_size}. "
    )


if __name__ == "__main__":
    fire.Fire(random_generate)