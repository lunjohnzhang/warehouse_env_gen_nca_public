"""Provides a generic function for executing maze envs."""
import logging
import random
import time
import traceback

import numpy as np

from env_search.maze.maze_result import MazeResult
from env_search.utils.worker_state import get_maze_module, get_maze_rl_agent_func

logger = logging.getLogger(__name__)


def run_maze(level: np.ndarray,
             n_evals: int,
             seed: int,
             eval_kwargs=None) -> MazeResult:
    """Grabs the maze module and evaluates level n_evals times."""
    start = time.time()
    eval_kwargs = {} if eval_kwargs is None else eval_kwargs

    logger.info("seeding global randomness")
    np.random.seed(seed // np.int32(4))
    random.seed(seed // np.int32(2))

    logger.info("run maze with %d n_evals and seed %d", n_evals, seed)
    maze_module = get_maze_module()
    rl_agent_func = get_maze_rl_agent_func()

    try:
        result = maze_module.evaluate(level=level,
                                      n_evals=n_evals,
                                      seed=seed,
                                      rl_agent_func=rl_agent_func)
    except TimeoutError as e:
        logger.warning(f"evaluate failed")
        logger.info(f"The level was {level}")
        result = MazeResult(
            failed=True,
            log_message=f"Evaluate failed with following error\n"
            f"{''.join(traceback.TracebackException.from_exception(e).format())}\n"
            f"Level was {level}")

    logger.info("run_maze done after %f sec", time.time() - start)

    result.maze_metadata = result.maze_metadata or {}
    if result.maze_metadata.get("level") is None:
        result.maze_metadata["level"] = level
    return result
