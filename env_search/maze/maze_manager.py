"""Provides MazeManager."""
import logging
from pathlib import Path
from typing import List, Tuple

import gin
import torch
import numpy as np
from dask.distributed import Client
from tqdm import tqdm

from env_search.device import DEVICE
from env_search.maze.agents.rl_agent import RLAgentConfig
from env_search.maze.emulation_model.buffer import Experience
from env_search.maze.emulation_model.aug_buffer import AugExperience
from env_search.maze.emulation_model.emulation_model import MazeEmulationModel
from env_search.maze.module import MazeModule, MazeConfig
from env_search.maze.maze_result import MazeResult
from env_search.maze.run import run_maze
from env_search.maze.generator.nca_generator import MazeNCA
from env_search.maze.level import MazeLevel
from env_search.utils import read_in_maze_map
from env_search.utils.worker_state import init_maze_module, init_maze_rl_agent_func

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["client", "rng"])
class MazeManager:
    """Manager for the maze environments.

    Args:
        client: Dask client for distributed compute.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        n_evals: Number of times to evaluate each solution during real
            evaluation.
        lvl_width: Width of the level.
        lvl_height: Height of the level.
        num_objects: Number of objects in the level to generate.
    """

    def __init__(
        self,
        client: Client,
        rng: np.random.Generator = None,
        n_evals: int = gin.REQUIRED,
        lvl_width: int = gin.REQUIRED,
        lvl_height: int = gin.REQUIRED,
        num_objects: int = gin.REQUIRED,
        is_nca: bool = gin.REQUIRED,
        nca_iter: int = gin.REQUIRED,
        seed_env_path: str = gin.REQUIRED,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        self.n_evals = n_evals
        self.lvl_width = lvl_width
        self.lvl_height = lvl_height
        self.num_objects = num_objects

        # Set up a module locally and on workers. During evaluations,
        # run_maze retrieves this module and uses it to evaluate the
        # function. Configuration is done with gin (i.e. the params are in the
        # config file).
        self.module = MazeModule(config := MazeConfig())
        client.register_worker_callbacks(lambda: init_maze_module(config))
        rl_agent_conf = RLAgentConfig()
        client.register_worker_callbacks(
            lambda: init_maze_rl_agent_func(rl_agent_conf))

        self.emulation_model = None

        # NCA related
        self.is_nca = is_nca
        self.nca_iter = nca_iter
        seed_map_str, _ = read_in_maze_map(seed_env_path)
        seed_map_int = MazeLevel.str_to_number(seed_map_str)
        self.seed_map_torch = torch.tensor(
            seed_map_int[np.newaxis, :, :],
            device=DEVICE,
        )

    def em_init(self,
                seed: int,
                pickle_path: Path = None,
                pytorch_path: Path = None):
        """Initialize the emulation model and optionally load from saved state.

        Args:
            seed: Random seed to use.
            pickle_path: Path to the saved emulation model data (optional).
            pytorch_path: Path to the saved emulation model network (optional).
        """
        self.emulation_model = MazeEmulationModel(seed=seed + 420)
        if pickle_path is not None:
            self.emulation_model.load(pickle_path, pytorch_path)
        logger.info("Emulation Model: %s", self.emulation_model)

    def get_initial_sols(self, size: Tuple):
        """Returns random solutions with the given size.

        Args:
            size: Tuple with (n_solutions, sol_size).

        Returns:
            Randomly generated solutions.
        """
        return self.rng.integers(self.num_objects, size=size), None

    def em_train(self):
        self.emulation_model.train()

    def emulation_pipeline(self, sols):
        """Pipeline that takes solutions and uses the emulation model to predict
        the objective and measures.

        Args:
            sols: Emitted solutions.

        Returns:
            lvls: Generated levels.
            objs: Predicted objective values.
            measures: Predicted measure values.
            success_mask: Array of size `len(lvls)`. An element in the array is
                False if some part of the prediction pipeline failed for the
                corresponding solution.
        """
        n_lvls = len(sols)
        lvls = np.array(sols).reshape(
            (n_lvls, self.lvl_height, self.lvl_width)).astype(int)
        success_mask = np.ones(len(lvls), dtype=bool)
        objs, measures = self.emulation_model.predict(lvls)
        return lvls, objs, measures, success_mask

    def eval_pipeline(self, sols, parent_sols=None, batch_idx=None):
        """Pipeline that takes a solution and evaluates it.

        Args:
            sols: Emitted solution.

        Returns:
            Results of the evaluation.
        """
        n_lvls = len(sols)

        # For NCA, use NCA model to generate actual lvls
        if self.is_nca:
            warehouseNCA = MazeNCA().to(DEVICE)
            lvls = []
            for i in tqdm(range(n_lvls)):
                warehouseNCA.set_params(sols[i])
                lvl, _ = warehouseNCA.generate(
                    self.seed_map_torch,
                    n_iter=self.nca_iter,
                )
                lvl = lvl.squeeze().cpu().numpy()
                lvls.append(lvl)
            lvls = np.array(lvls).reshape(
                (n_lvls, self.lvl_height, self.lvl_width)).astype(int)
        else:
            lvls = np.array(sols).reshape(
                (n_lvls, self.lvl_height, self.lvl_width)).astype(int)

        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                             size=len(sols),
                                             endpoint=True)
        futures = [
            self.client.submit(
                run_maze,
                lvl,
                self.n_evals,
                seed,
                pure=False,
            ) for lvl, seed in zip(lvls, evaluation_seeds)
        ]
        logger.info("Collecting evaluations")
        results: List[MazeResult] = self.client.gather(futures)

        return results

    def add_experience(self, sol, result):
        """Add required experience to the emulation model based on the solution
        and the results.

        Args:
            sol: Emitted solution.
            result: Evaluation result.
        """
        obj = result.agg_obj
        meas = result.agg_measures
        input_lvl = result.maze_metadata["level"]
        if self.emulation_model.pre_network is not None:
            self.emulation_model.add(
                AugExperience(sol, input_lvl, obj, meas,
                              result.maze_metadata["aug_level"]))
        else:
            self.emulation_model.add(Experience(sol, input_lvl, obj, meas))

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed levels.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution": sol,
            "level": result.maze_metadata["level"],
            "log_message": result.log_message,
        }
        return failed_level_info
