"""Functions for managing worker state.

In general, one uses these by first calling init_* or set_* to create the
attribute, then calling get_* to retrieve the corresponding value.
"""
from functools import partial

from dask.distributed import get_worker

from env_search.warehouse.module import WarehouseConfig, WarehouseModule
from env_search.maze.agents.rl_agent import RLAgentConfig, RLAgent
from env_search.maze.module import MazeConfig, MazeModule
from env_search.manufacture.module import ManufactureConfig, ManufactureModule


#
# Generic
#


def set_worker_state(key: str, val: object):
    """Sets worker_state[key] = val"""
    worker = get_worker()
    setattr(worker, key, val)


def get_worker_state(key: str) -> object:
    """Retrieves worker_state[key]"""
    worker = get_worker()
    return getattr(worker, key)


#
# Warehouse module
#

WAREHOUSE_MOD_ATTR = "warehouse_module"


def init_warehouse_module(config: WarehouseConfig):
    """Initializes this worker's warehouse module."""
    set_worker_state(WAREHOUSE_MOD_ATTR, WarehouseModule(config))


def get_warehouse_module() -> WarehouseModule:
    """Retrieves this worker's warehouse module."""
    return get_worker_state(WAREHOUSE_MOD_ATTR)


#
# Manufacture module
#

MANUFACTURE_MOD_ATTR = "manufacture_module"


def init_manufacture_module(config: ManufactureConfig):
    """Initializes this worker's manufacture module."""
    set_worker_state(MANUFACTURE_MOD_ATTR, ManufactureModule(config))


def get_manufacture_module() -> ManufactureModule:
    """Retrieves this worker's manufacture module."""
    return get_worker_state(MANUFACTURE_MOD_ATTR)


#
# Maze module
#

MAZE_MOD_ATTR = "maze_module"


def init_maze_module(config: MazeConfig):
    """Initializes this worker's maze module."""
    set_worker_state(MAZE_MOD_ATTR, MazeModule(config))


def get_maze_module() -> MazeModule:
    """Retrieves this worker's maze module."""
    return get_worker_state(MAZE_MOD_ATTR)


#
# Maze RL agent
#

MAZE_RL_AGENT_MOD_ATTR = "maze_rl_agent"


def init_maze_rl_agent_func(config: RLAgentConfig):
    """Initializes this worker's maze module."""
    set_worker_state(MAZE_RL_AGENT_MOD_ATTR, partial(RLAgent, config=config))


def get_maze_rl_agent_func() -> callable:
    """Retrieves this worker's maze rl agent."""
    return get_worker_state(MAZE_RL_AGENT_MOD_ATTR)