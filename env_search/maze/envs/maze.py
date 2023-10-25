# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Code from https://github.com/ucl-dark/paired.

Implements single-agent manually generated Maze environments.

Humans provide a bit map to describe the position of walls, the starting
location of the agent, and the goal location.
"""
import logging

import gym_minigrid.minigrid as minigrid
import numpy as np

from .multigrid import MultiGridEnv, Grid

logger = logging.getLogger(__name__)


class MazeEnv(MultiGridEnv):
    """Single-agent maze environment specified via a bit map."""

    def __init__(
        self,
        agent_view_size=5,
        minigrid_mode=True,
        max_steps=None,
        bit_map=None,
        start_pos=None,
        goal_pos=None,
        size=15,
    ):
        default_agent_start_x = 7
        default_agent_start_y = 1
        default_goal_start_x = 7
        default_goal_start_y = 13
        self.start_pos = (np.array([
            default_agent_start_x, default_agent_start_y
        ]) if start_pos is None else start_pos)
        self.goal_pos = ((default_goal_start_x, default_goal_start_y)
                         if goal_pos is None else goal_pos)

        if max_steps is None:
            max_steps = 2 * size * size

        if bit_map is not None:
            bit_map = np.array(bit_map)
            if bit_map.shape != (size - 2, size - 2):
                logger.warning(
                    "Error! Bit map shape does not match size. Using default maze."
                )
                bit_map = None

        if bit_map is None:
            self.bit_map = np.array([
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            ])
        else:
            self.bit_map = bit_map

        super().__init__(
            n_agents=1,
            grid_size=size,
            agent_view_size=agent_view_size,
            max_steps=max_steps,
            see_through_walls=True,  # Set this to True for maximum speed
            minigrid_mode=minigrid_mode,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Goal
        self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])

        # Agent
        self.place_agent_at_pos(0, self.start_pos)

        # Walls
        for x in range(self.bit_map.shape[0]):
            for y in range(self.bit_map.shape[1]):
                if self.bit_map[y, x]:
                    # Add an offset of 1 for the outer walls
                    self.put_obj(minigrid.Wall(), x + 1, y + 1)
