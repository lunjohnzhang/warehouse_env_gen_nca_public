from functools import partial

import fire
import numpy as np
from scipy.sparse import csgraph
from skimage.segmentation import flood_fill

from env_search.maze.agents.rl_agent import RLAgent, RLAgentConfig
from env_search.maze.envs.maze import MazeEnv
from env_search.maze.level import MazeLevel, OBJ_TYPES_TO_INT
from env_search.maze.module import MazeModule


class LabyrinthEnv(MazeEnv):
    """A short but non-optimal path is 118 moves."""

    def __init__(self):
        # positions go col, row
        start_pos = np.array([1, 13])
        goal_pos = np.array([7, 7])
        bit_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        super().__init__(size=15,
                         bit_map=bit_map,
                         start_pos=start_pos,
                         goal_pos=goal_pos)


class LargeCorridorEnv(MazeEnv):
    """A long backtracking env."""

    def __init__(self):
        # positions go col, row and indexing starts at 1
        start_pos = np.array([1, 10])
        row = np.random.choice([9, 11])
        col = np.random.choice([3, 5, 7, 9, 11, 13, 15, 17])
        goal_pos = np.array([col, row])
        bit_map = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        super().__init__(size=21,
                         bit_map=bit_map,
                         start_pos=start_pos,
                         goal_pos=goal_pos)


class CustomEnv(MazeEnv):
    """Custom env."""

    def __init__(self):
        # positions go col, row and indexing starts at 1
        start_pos = np.array([9, 4])
        goal_pos = np.array([7, 8])
        bit_map = np.array([[1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                            [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
                            [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
                            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]])
        super().__init__(size=18,
                         bit_map=bit_map,
                         start_pos=start_pos,
                         goal_pos=goal_pos)


def main():
    rng = np.random.default_rng(24)

    while True:
        level = rng.integers(2, size=(16, 16))
        print("Generated level:")
        print(MazeLevel(level).to_str())
        print("Bit map:")
        print(level.tolist())
        adj = MazeModule._get_adj(level)

        # Find the best distances
        dist, predecessors = csgraph.floyd_warshall(adj,
                                                    return_predecessors=True)
        dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

        if dist.max() >= 1:
            print(f"Optimal path length: {dist.max()}")
            # Label the start and the end point
            endpoints = np.unravel_index(dist.argmax(), dist.shape)
            start_cell, end_cell = zip(
                *np.unravel_index(endpoints, level.shape))

            endpoint_level = level.copy()
            endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
            endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

            break

    # Offset start, goal to account for the added outer walls
    start_pos = (start_cell[1] + 1, start_cell[0] + 1)
    goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
    print(f"Start: {start_pos}; End: {goal_pos}")
    env_func = partial(MazeEnv,
                       size=level.shape[0] + 2,
                       bit_map=level,
                       start_pos=start_pos,
                       goal_pos=goal_pos)
    # env_func = LabyrinthEnv
    # env_func = LargeCorridorEnv
    # env_func = CustomEnv
    rl_agent_conf = RLAgentConfig(recurrent_hidden_size=256,
                                  model_path="accel_seed_1/model_20000.tar")
    rl_agent = RLAgent(env_func, n_evals=100, config=rl_agent_conf)
    rl_result = rl_agent.eval_and_track(level_shape=level.shape)
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

    print(f"Path lengths: {rl_result.path_lengths}")
    print(f"Fails: {rl_result.failed_list}")
    print(f"Left turns: {rl_result.n_left_turns}")
    print(f"Right turns: {rl_result.n_right_turns}")
    print(f"Repeated cells: {rl_result.n_repeated_cells}")
    print(f"Frac explored: {frac_explored_cells}")
    print(f"Aug shape: {rl_result.aug_level.shape}")


if __name__ == '__main__':
    fire.Fire(main)
