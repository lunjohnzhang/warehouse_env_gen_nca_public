"""Provides ManufactureManager."""
import logging
from pathlib import Path
from typing import List, Tuple

import time
import gin
import numpy as np
import copy
import torch
from dask.distributed import Client
from logdir import LogDir
from tqdm import tqdm

from env_search.device import DEVICE
from env_search.manufacture.emulation_model.buffer import Experience
from env_search.manufacture.emulation_model.aug_buffer import AugExperience
from env_search.manufacture.emulation_model.double_aug_buffer import DoubleAugExperience
from env_search.manufacture.emulation_model.emulation_model import ManufactureEmulationModel
from env_search.manufacture.emulation_model.networks import (
    ManufactureAugResnetOccupancy, ManufactureAugResnetRepairedMapAndOccupancy)
from env_search.manufacture.module import (ManufactureModule, ManufactureConfig)
from env_search.manufacture.generator.nca_generator import ManufactureNCA
from env_search.manufacture.run import (run_manufacture, repair_manufacture,
                                        process_manufacture_eval_result)
from env_search.utils.worker_state import init_manufacture_module

from env_search.utils import (manufacture_obj_types, MIN_SCORE,
                              manufacture_env_number2str,
                              manufacture_env_str2number, format_env_str,
                              read_in_manufacture_map, flip_tiles)

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["client", "rng"])
class ManufactureManager:
    """Manager for the manufacture environments.

    Args:
        client: Dask client for distributed compute.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        n_evals: Number of times to evaluate each solution during real
            evaluation.
        lvl_width: Width of the level.
        lvl_height: Height of the level.
        num_objects: Number of objects in the level to generate.
        min_n_shelf (int): min number of shelves
        max_n_shelf (int): max number of shelves
        w_mode (bool): whether to run with w_mode, which replace 'r' with 'w' in
                       generated map layouts, where 'w' is a workstation.
                       Under w_mode, robots will start from endpoints and their
                       tasks will alternate between endpoints and workstations.
        n_endpt (int): number of endpoint around each obstacle
        agent_num (int): number of drives
    """

    def __init__(
        self,
        client: Client,
        logdir: LogDir,
        rng: np.random.Generator = None,
        n_evals: int = gin.REQUIRED,
        lvl_width: int = gin.REQUIRED,
        lvl_height: int = gin.REQUIRED,
        num_objects: int = gin.REQUIRED,
        min_n_shelf: int = gin.REQUIRED,
        max_n_shelf: int = gin.REQUIRED,
        # w_mode: bool = gin.REQUIRED,
        n_endpt: bool = gin.REQUIRED,
        agent_num: int = gin.REQUIRED,
        is_nca: bool = gin.REQUIRED,
        nca_iter: int = gin.REQUIRED,
        seed_env_path: str = gin.REQUIRED,
        n_worker_repair: int = -1,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        self.n_evals = n_evals
        self.eval_batch_idx = 0  # index of each batch of evaluation

        self.logdir = logdir

        self.lvl_width = lvl_width
        self.lvl_height = lvl_height

        self.num_objects = num_objects

        self.min_n_shelf = min_n_shelf
        self.max_n_shelf = max_n_shelf

        self.n_endpt = n_endpt
        self.agent_num = agent_num

        self.n_worker_repair = n_worker_repair

        # Set up a module locally and on workers. During evaluations,
        # repair_and_run_manufacture retrieves this module and uses it to
        # evaluate the function. Configuration is done with gin (i.e. the
        # params are in the config file).
        self.module = ManufactureModule(config := ManufactureConfig())
        client.register_worker_callbacks(
            lambda: init_manufacture_module(config))

        self.emulation_model = None

        # NCA related
        self.is_nca = is_nca
        self.nca_iter = nca_iter
        seed_map_str, _ = read_in_manufacture_map(seed_env_path)
        seed_map_int = manufacture_env_str2number(seed_map_str)
        self.seed_map_torch = torch.tensor(
            seed_map_int[np.newaxis, :, :],
            device=DEVICE,
        )

        # Runtime
        self.repair_runtime = 0
        self.sim_runtime = 0

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
        self.emulation_model = ManufactureEmulationModel(seed=seed + 420)
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
        batch_size, solution_dim = size
        sols = self.rng.integers(self.num_objects,
                                 size=(batch_size, solution_dim))

        return np.array(sols), None

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
        n_maps = len(sols)
        maps = np.array(sols).reshape(
            (n_maps, self.lvl_height, self.lvl_width)).astype(int)

        success_mask = np.ones(len(maps), dtype=bool)
        objs, measures = self.emulation_model.predict(maps)
        return maps, objs, measures, success_mask

    # def eval_pipeline(self, sols, parent_sols=None, batch_idx=None):
    #     """Pipeline that takes a solution and evaluates it.

    #     Args:
    #         sols: Emitted solution.
    #         parent_sols: Parent solution of sols.

    #     Returns:
    #         Results of the evaluation.
    #     """
    #     n_lvls = len(sols)

    #     # For NCA, use NCA model to generate actual lvls
    #     if self.is_nca:
    #         nca_start_time = time.time()
    #         manufactureNCA = ManufactureNCA().to(DEVICE)
    #         lvls = []
    #         for i in tqdm(range(n_lvls)):
    #             manufactureNCA.set_params(sols[i])
    #             lvl, _ = manufactureNCA.generate(
    #                 self.seed_map_torch,
    #                 n_iter=self.nca_iter,
    #             )
    #             lvl = lvl.squeeze().cpu().numpy()
    #             lvls.append(lvl)
    #         lvls = np.array(lvls).reshape(
    #             (n_lvls, self.lvl_height, self.lvl_width)).astype(int)
    #         nca_time_lapsed = time.time() - nca_start_time
    #         logger.info(f"NCA takes {round(nca_time_lapsed, 3)} seconds")
    #     else:
    #         lvls = np.array(sols).reshape(
    #             (n_lvls, self.lvl_height, self.lvl_width)).astype(int)

    #     if parent_sols is not None:
    #         parent_lvls = np.array(parent_sols)
    #     else:
    #         parent_lvls = [None] * n_lvls

    #     # Make each solution evaluation have a different seed. Note that we
    #     # assign seeds to solutions rather than workers, which means that we
    #     # are agnostic to worker configuration.
    #     evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
    #                                          size=n_lvls,
    #                                          endpoint=True)

    #     # Split repair and evaluate.
    #     # Since evaluation might take a lot longer than repair, and each
    #     # evaluation might includes several simulations, we want to distribute
    #     # all the simulations to the workers instead of evaluations to fully
    #     # exploit the available compute

    #     # First, repair the maps
    #     # Use part of the workers for repairing if necessary
    #     if self.n_worker_repair == -1:
    #         repair_workers = None
    #     else:
    #         worker_info = self.client.scheduler_info()['workers']
    #         worker_addresses = list(worker_info.keys())
    #         repair_workers = worker_addresses[:self.n_worker_repair]
    #         assert len(repair_workers) > 0
    #     repair_futures = [
    #         self.client.submit(
    #             repair_manufacture,
    #             workers=repair_workers,
    #             map=lvl,
    #             parent_map=parent_lvl,
    #             sim_seed=seed,
    #             repair_seed=self.seed,
    #             min_n_shelf=self.min_n_shelf,
    #             max_n_shelf=self.max_n_shelf,
    #         ) for lvl, parent_lvl, seed in zip(lvls, parent_lvls,
    #                                            evaluation_seeds)
    #     ]

    #     # Update batch id
    #     if batch_idx is None:
    #         batch_idx = self.eval_batch_idx
    #     eval_logdir = self.logdir.pdir(f"evaluations/eval_batch_{batch_idx}")
    #     self.eval_batch_idx += 1

    #     sim_futures = []
    #     map_ids = np.arange(n_lvls)
    #     n_evals = self.n_evals
    #     max_n_shelf = self.max_n_shelf
    #     agent_num = self.agent_num
    #     # For each repair future, submit n_eval sim futures
    #     for map_id, repair_future, eval_seed in zip(
    #             map_ids,
    #             repair_futures,
    #             evaluation_seeds,
    #     ):
    #         for j in range(self.n_evals):
    #             sim_seed = eval_seed + j
    #             sim_future = self.client.submit(
    #                 # Use a lambda function to extract map_json from the
    #                 # repair_manufacture output
    #                 lambda repair_result: run_manufacture(
    #                     repair_result[0],
    #                     eval_logdir=eval_logdir,
    #                     sim_seed=sim_seed,
    #                     agentNum=agent_num,
    #                     map_id=map_id,
    #                     eval_id=j,
    #                 ),
    #                 repair_future,  # Pass in future to create a dask workflow
    #             )
    #             sim_futures.append(sim_future)

    #     # Postprocess the result
    #     process_futures = []

    #     for i, repair_future in enumerate(repair_futures):
    #         curr_sim_futures = sim_futures[i * self.n_evals:(i + 1) *
    #                                        self.n_evals]

    #         process_future = self.client.submit(
    #             lambda repair_result, sim_results:
    #             process_manufacture_eval_result(
    #                 sim_results,
    #                 n_evals=n_evals,
    #                 map_np_unrepaired=repair_result[1],
    #                 map_np_repaired=repair_result[2],
    #                 max_n_shelf=max_n_shelf,
    #                 map_id=i,  # should be the same as map_id
    #             ),
    #             repair_future,
    #             curr_sim_futures,
    #         )
    #         process_futures.append(process_future)

    #     results = self.client.gather(process_futures)

    #     return results


    def eval_pipeline(self, sols, parent_sols=None, batch_idx=None):
        """Pipeline that takes a solution and evaluates it.
        Args:
            sols: Emitted solution.
            parent_sols: Parent solution of sols.
        Returns:
            Results of the evaluation.
        """
        n_lvls = len(sols)

        # For NCA, use NCA model to generate actual lvls
        if self.is_nca:
            nca_start_time = time.time()
            manufactureNCA = ManufactureNCA().to(DEVICE)
            lvls = []
            for i in tqdm(range(n_lvls)):
                manufactureNCA.set_params(sols[i])
                lvl, _ = manufactureNCA.generate(
                    self.seed_map_torch,
                    n_iter=self.nca_iter,
                )
                lvl = lvl.squeeze().cpu().numpy()
                lvls.append(lvl)
            lvls = np.array(lvls).reshape(
                (n_lvls, self.lvl_height, self.lvl_width)).astype(int)
            nca_time_lapsed = time.time() - nca_start_time
            logger.info(f"NCA takes {round(nca_time_lapsed, 3)} seconds")
        else:
            lvls = np.array(sols).reshape(
                (n_lvls, self.lvl_height, self.lvl_width)).astype(int)

        if parent_sols is not None:
            parent_lvls = np.array(parent_sols)
        else:
            parent_lvls = [None] * n_lvls

        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                             size=n_lvls,
                                             endpoint=True)

        # Split repair and evaluate.
        # Since evaluation might take a lot longer than repair, and each
        # evaluation might includes several simulations, we want to distribute
        # all the simulations to the workers instead of evaluations to fully
        # exploit the available compute

        # First, repair the maps
        # Use part of the workers for repairing if necessary
        if self.n_worker_repair == -1:
            repair_workers = None
        else:
            worker_info = self.client.scheduler_info()['workers']
            worker_addresses = list(worker_info.keys())
            repair_workers = worker_addresses[:self.n_worker_repair]
            assert len(repair_workers) > 0

        repair_start_time = time.time()
        repair_futures = [
            self.client.submit(
                repair_manufacture,
                workers=repair_workers,
                map=lvl,
                parent_map=parent_lvl,
                sim_seed=seed,
                repair_seed=self.seed,
                min_n_shelf=self.min_n_shelf,
                max_n_shelf=self.max_n_shelf,
                agent_num=self.agent_num,
            ) for lvl, parent_lvl, seed in zip(lvls, parent_lvls,
                                               evaluation_seeds)
        ]

        repair_results = self.client.gather(repair_futures)
        self.repair_runtime += time.time() - repair_start_time

        # Based on number of simulations (n_evals), create maps and
        # corresponding variables to simulate
        map_jsons_sim = []
        map_np_unrepaired_sim = []
        map_np_repaired_sim = []
        maps_id_sim = []
        maps_eval_seed_sim = []
        eval_id_sim = []
        map_ids = np.arange(n_lvls)
        for map_id, repair_result, eval_seed in zip(
                map_ids,
                repair_results,
                evaluation_seeds,
        ):
            (
                map_json,
                map_np_unrepaired,
                map_np_repaired,
            ) = repair_result
            for j in range(self.n_evals):
                map_jsons_sim.append(copy.deepcopy(map_json))
                map_np_unrepaired_sim.append(copy.deepcopy(map_np_unrepaired))
                map_np_repaired_sim.append(copy.deepcopy(map_np_repaired))
                maps_id_sim.append(map_id)
                maps_eval_seed_sim.append(eval_seed + j)
                eval_id_sim.append(j)

        # Then, evaluate the maps
        if batch_idx is None:
            batch_idx = self.eval_batch_idx
        eval_logdir = self.logdir.pdir(f"evaluations/eval_batch_{batch_idx}")
        self.eval_batch_idx += 1
        sim_start_time = time.time()
        sim_futures = [
            self.client.submit(
                run_manufacture,
                map_json=map_json,
                eval_logdir=eval_logdir,
                sim_seed=sim_seed,
                agentNum=self.agent_num,
                map_id=map_id,
                eval_id=eval_id,
            ) for (
                map_json,
                sim_seed,
                map_id,
                eval_id,
            ) in zip(
                map_jsons_sim,
                maps_eval_seed_sim,
                maps_id_sim,
                eval_id_sim,
            )
        ]
        logger.info("Collecting evaluations")
        results_json = self.client.gather(sim_futures)
        self.sim_runtime += time.time() - sim_start_time

        results_json_sorted = []
        for i in range(len(sols)):
            curr_eval_results = []
            for j in range(self.n_evals):
                curr_eval_results.append(results_json[i * self.n_evals + j])
            results_json_sorted.append(curr_eval_results)

        logger.info("Processing eval results")

        process_futures = [
            self.client.submit(
                process_manufacture_eval_result,
                curr_result_json=curr_result_json,
                n_evals=self.n_evals,
                map_np_unrepaired=map_np_unrepaired_sim[map_id * self.n_evals],
                map_np_repaired=map_np_repaired_sim[map_id * self.n_evals],
                max_n_shelf=self.max_n_shelf,
                map_id=map_id,
            ) for (
                curr_result_json,
                map_id,
            ) in zip(
                results_json_sorted,
                map_ids,
            )
        ]
        results = self.client.gather(process_futures)

        # results = []
        # for (curr_result_json, map_id) in zip(results_json_sorted, map_ids):
        #     try:
        #         result = self.module.process_eval_result(
        #             curr_result_json,
        #             n_evals=self.n_evals,
        #             map_np_unrepaired=map_np_unrepaired_sim[map_id * self.n_evals],
        #             map_np_repaired=map_np_repaired_sim[map_id * self.n_evals],
        #             max_n_shelf=self.max_n_shelf,
        #             map_id=map_id,
        #         )
        #     except TypeError:
        #         breakpoint()
        #     results.append(result)

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
        input_lvl = result.manufacture_metadata["map_int_unrepaired"]
        repaired_lvl = result.manufacture_metadata["map_int"]

        if self.emulation_model.pre_network is not None:
            # Mean of tile usage over n_evals
            avg_tile_usage = np.mean(result.manufacture_metadata["tile_usage"],
                                     axis=0)
            if isinstance(self.emulation_model.pre_network,
                          ManufactureAugResnetOccupancy):
                self.emulation_model.add(
                    AugExperience(sol, input_lvl, obj, meas, avg_tile_usage))
            elif isinstance(self.emulation_model.pre_network,
                            ManufactureAugResnetRepairedMapAndOccupancy):
                self.emulation_model.add(
                    DoubleAugExperience(sol, input_lvl, obj, meas,
                                        avg_tile_usage, repaired_lvl))
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
            "map_int_unrepaired":
            result.manufacture_metadata["map_int_unrepaired"],
            "log_message": result.log_message,
        }
        return failed_level_info
