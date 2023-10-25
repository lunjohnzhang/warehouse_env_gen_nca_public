from functools import partial

import os
import time
import fire
import numpy as np
import cv2
import copy
from logdir import LogDir
from scipy.sparse import csgraph
from skimage.segmentation import flood_fill

from env_search import LOG_DIR
from env_search.maze.agents.rl_agent import RLAgent, RLAgentConfig
from env_search.maze.envs.maze import MazeEnv
from env_search.maze.level import MazeLevel, OBJ_TYPES_TO_INT
from env_search.maze.module import MazeModule
from env_search.utils import read_in_maze_map


def run_single_rl_agent(
    level_filepath: str,
    framerate: int = 24,
    n_evals: int = 100,
    render: bool = False,
    logdir: LogDir = None,
):
    level, maze_name = read_in_maze_map(level_filepath)
    level_str = copy.deepcopy(level)
    level = MazeLevel.str_to_number(level).astype(int)

    print("Generated level:")
    print(MazeLevel(level).to_str())
    print("Bit map:")
    print(level.tolist())
    adj = MazeModule._get_adj(level)

    # Find the best distances
    dist, predecessors = csgraph.floyd_warshall(adj, return_predecessors=True)
    dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

    print(f"Optimal path length: {dist.max()}")
    # Label the start and the end point
    endpoints = np.unravel_index(dist.argmax(), dist.shape)
    start_cell, end_cell = zip(*np.unravel_index(endpoints, level.shape))

    endpoint_level = level.copy()
    endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
    endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

    # Offset start, goal to account for the added outer walls
    start_pos = (start_cell[1] + 1, start_cell[0] + 1)
    goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
    print(f"Start: {start_pos}; End: {goal_pos}")
    env_func = partial(MazeEnv,
                       size=level.shape[0] + 2,
                       bit_map=level,
                       start_pos=start_pos,
                       goal_pos=goal_pos)

    rl_agent_conf = RLAgentConfig(recurrent_hidden_size=256,
                                  model_path="accel_seed_1/model_20000.tar",
                                  n_envs=1)
    rl_agent = RLAgent(env_func, n_evals=n_evals, config=rl_agent_conf)
    frame = 0

    if logdir is None:
        logdir = LogDir(maze_name, rootdir=LOG_DIR)

    if not render:
        render_func = None
    else:
        video_dir = logdir.pdir(f"maze_viz_{maze_name}", touch=True)

        def render_func(vec_env):
            nonlocal frame
            img = vec_env.render(mode="rgb_array")
            name = video_dir / f"frame-{frame:08d}.png"
            cv2.imwrite(str(name), img)
            if frame % 100 == 0:
                print(f"Saved frame {frame}")
            frame += 1

    rl_result = rl_agent.eval_and_track(level_shape=level.shape,
                                        render_func=render_func)
    if render:
        print("Assembling video with ffmpeg")
        os.system(f"""\
ffmpeg -an -r {framerate} -i "{video_dir / 'frame-%*.png'}" \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3 \
    {logdir.file(f'maze_viz_{maze_name}/solution.mp4')} \
    -y \
""")
    # objs, aug_level, n_left_turns, n_right_turns = rl_agent.eval_and_track(
    #     level_shape=level.shape,
    #     obj_type="path_length",
    #     aug_type="agent_occupancy")
    # objs, aug_level, n_left_turns, n_right_turns = rl_agent.eval_and_track(
    #     level_shape=level.shape, obj_type="fail_rate", aug_type="turns")
    # objs = rl_agent.eval_and_track(level_shape=level.shape)

    flood_fill_level = flood_fill(level, start_cell, -1, connectivity=1)
    n_reachable_cells = np.sum(flood_fill_level == -1)
    n_explored_cells = np.sum(rl_result.aug_level > 0)
    frac_explored_cells = n_explored_cells / n_reachable_cells
    success_rate = 1 - np.sum(rl_result.failed_list) / n_evals

    print(f"Path lengths: {rl_result.path_lengths}")
    print(f"Fails: {rl_result.failed_list}")
    print(f"Success rate: {round(success_rate, 2)}")
    print(f"Left turns: {rl_result.n_left_turns}")
    print(f"Right turns: {rl_result.n_right_turns}")
    print(f"Repeated cells: {rl_result.n_repeated_cells}")
    print(f"Frac explored: {frac_explored_cells}")
    print(f"Aug shape: {rl_result.aug_level.shape}")

    result = {
        "name": maze_name,
        "path_lengths": rl_result.path_lengths.tolist(),
        "fails": rl_result.failed_list.tolist(),
        "success_rate": success_rate,
        "left_turns": rl_result.n_left_turns.tolist(),
        "right_turns": rl_result.n_right_turns.tolist(),
        "repeated_cells": rl_result.n_repeated_cells,
        "frac_explored": frac_explored_cells,
        "aug_shape": rl_result.aug_level.shape,
        "layout": level_str,
    }

    return result


if __name__ == '__main__':
    fire.Fire(run_single_rl_agent)
