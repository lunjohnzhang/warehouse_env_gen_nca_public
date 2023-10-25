"""ManufactureConfig and ManufactureModule.

Usage:
    # Run as a script to demo the ManufactureModule.
    python env_search/manufacture/module.py
"""

import os
import gin
import copy
import json
import time
import fire
import logging
import pathlib
import warnings
import warehouse_sim  # type: ignore # ignore pylance warning
import numpy as np
import multiprocessing
import shutil

from scipy.stats import entropy
from pprint import pprint
from typing import List
from dataclasses import dataclass
from itertools import repeat, product
from typing import Collection, Optional
from queue import Queue
from typing import Collection
from env_search import LOG_DIR
from env_search.utils.logging import setup_logging
from env_search.manufacture.milp_repair import repair_env
from env_search.manufacture.manufacture_result import (ManufactureResult,
                                                       ManufactureMetadata)
from env_search.utils import (manufacture_obj_types, manufacture_env_number2str,
                              manufacture_env_str2number, format_env_str,
                              read_in_manufacture_map, flip_tiles, MIN_SCORE)

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class ManufactureConfig:
    """
    Config manufacture simulation

    Args:
        measure_names (list[str]): list of names of measures
        aggregation_type (str): aggregation over `n_evals` results
        scenario (str): scenario (SORTING, KIVA, ONLINE, BEE)
        task (str): input task file

        cutoffTime (int): cutoff time (seconds)
        screen (int): screen option (0: none; 1: results; 2:all)
        solver (str): solver (LRA, PBS, WHCA, ECBS)
        id (bool): independence detection
        single_agent_solver (str): single-agent solver (ASTAR, SIPP)
        lazyP (bool): use lazy priority
        simulation_time (int): run simulation
        simulation_window (int): call the planner every simulation_window
                                 timesteps
        travel_time_window (int): consider the traffic jams within the
                                  given window
        planning_window (int): the planner outputs plans with first
                                     planning_window timesteps collision-free
        potential_function (str): potential function (NONE, SOC, IC)
        potential_threshold (int): potential threshold
        rotation (bool): consider rotation
        robust (int): k-robust (for now, only work for PBS)
        CAT (bool): use conflict-avoidance table
        hold_endpoints (bool): Hold endpoints from Ma et al, AAMAS 2017
        dummy_paths (bool): Find dummy paths from Liu et al, AAMAS 2019
        prioritize_start (bool): Prioritize waiting at start locations
        suboptimal_bound (int): Suboptimal bound for ECBS
        log (bool): save the search trees (and the priority trees)
        test (bool): whether under testing mode.
        use_warm_up (bool): if True, will use the warm-up procedure. In
                            particular, for the initial population, the solution
                            returned from hamming distance objective will be
                            used. For mutated solutions, the solution of the
                            parent will be used.
        save_result (bool): Whether to allow C++ save the result of simulation
        save_solver (bool): Whether to allow C++ save the result of solver
        save_heuristics_table (bool): Whether to allow C++ save the result of
                                      heuristics table
        stop_at_traffic_jam (bool): whether stop the simulation at traffic jam
        obj_type (str): type of objective
                        ("throughput",
                         "throughput_plus_n_shelf",
                         "throughput_minus_hamming_dist")
    """
    # Measures.
    measure_names: Collection[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = gin.REQUIRED

    # Simulation
    scenario: str = gin.REQUIRED
    task: str = gin.REQUIRED
    cutoffTime: int = gin.REQUIRED
    screen: int = gin.REQUIRED
    solver: str = gin.REQUIRED
    id: bool = gin.REQUIRED
    single_agent_solver: str = gin.REQUIRED
    lazyP: bool = gin.REQUIRED
    simulation_time: int = gin.REQUIRED
    simulation_window: int = gin.REQUIRED
    travel_time_window: int = gin.REQUIRED
    planning_window: int = gin.REQUIRED
    potential_function: str = gin.REQUIRED
    potential_threshold: int = gin.REQUIRED
    rotation: bool = gin.REQUIRED
    robust: int = gin.REQUIRED
    CAT: bool = gin.REQUIRED
    hold_endpoints: bool = gin.REQUIRED
    dummy_paths: bool = gin.REQUIRED
    prioritize_start: bool = gin.REQUIRED
    suboptimal_bound: int = gin.REQUIRED
    log: bool = gin.REQUIRED
    test: bool = gin.REQUIRED
    use_warm_up: bool = gin.REQUIRED
    hamming_only: bool = gin.REQUIRED
    save_result: bool = gin.REQUIRED
    save_solver: bool = gin.REQUIRED
    save_heuristics_table: bool = gin.REQUIRED
    stop_at_traffic_jam: bool = gin.REQUIRED
    obj_type: str = gin.REQUIRED
    n_station_types: int = gin.REQUIRED
    station_wait_times: List[int] = gin.REQUIRED
    hamming_obj_weight: float = 1
    repair_n_threads: int = 1
    repair_timelimit: int = 60
    station_same_weight: int = 1
    OverallCutoffTime: int = None


class ManufactureModule:

    def __init__(self, config: ManufactureConfig):
        self.config = config

    def repair(
        self,
        map: np.ndarray,
        parent_map: np.ndarray,
        repair_seed: int,
        sim_seed: int,
        min_n_shelf: int,
        max_n_shelf: int,
        agent_num: int,
    ):
        map_np_unrepaired = copy.deepcopy(map)
        n_row, n_col = map_np_unrepaired.shape

        # Create json string for the map
        if self.config.scenario == "MANUFACTURE":
            if self.config.obj_type == "throughput_plus_n_shelf":
                assert max_n_shelf == min_n_shelf

            # Repair environment here
            format_env = format_env_str(
                manufacture_env_number2str(map_np_unrepaired))

            logger.info(f"Repairing generated environment:\n{format_env}")

            # Limit n_shelf?
            limit_n_shelf = True
            if self.config.obj_type == "throughput_plus_n_shelf":
                limit_n_shelf = False
            # Warm start schema
            warm_up_sols = None
            if self.config.use_warm_up:
                if parent_map is not None:
                    parent_env_str = format_env_str(
                        manufacture_env_number2str(parent_map))
                    logger.info(f"Parent warm up solution:\n{parent_env_str}")
                    warm_up_sols = [parent_map]
                # Get the solution from hamming distance objective
                hamming_repaired_env = repair_env(
                    map_np_unrepaired,
                    add_movement=False,
                    min_n_shelf=min_n_shelf,
                    max_n_shelf=max_n_shelf,
                    seed=repair_seed,
                    warm_envs_np=warm_up_sols,
                    limit_n_shelf=limit_n_shelf,
                    n_threads=self.config.repair_n_threads,
                    time_limit=self.config.repair_timelimit,
                    agent_num=agent_num,
                )
                ##############################################################
                ## For testing purpose, randomly force some layout to fail ###
                # rnd = np.random.rand()
                # if rnd > 0.5:
                #     hamming_repaired_env = None
                ##############################################################

                # If the repair is failed (which happens very rarely), we
                # return None and remember the unrepaired layout.
                if hamming_repaired_env is None:
                    failed_unrepaired_env = format_env_str(
                        manufacture_env_number2str(map_np_unrepaired))
                    logger.info(
                        f"Hamming repair failed! The layout is:\n{failed_unrepaired_env}"
                    )
                    return None, map_np_unrepaired, None
                hamming_warm_env_str = format_env_str(
                    manufacture_env_number2str(hamming_repaired_env))
                logger.info(
                    f"Hamming warm up solution:\n{hamming_warm_env_str}")

                if parent_map is None:
                    warm_up_sols = [hamming_repaired_env]
                else:
                    warm_up_sols = [hamming_repaired_env, parent_map]

            # If hamming only, we just use hamming_repaired_env as the result
            # env
            if self.config.hamming_only:
                map_np_repaired = hamming_repaired_env
            else:
                map_np_repaired = repair_env(
                    map_np_unrepaired,
                    add_movement=True,
                    warm_envs_np=warm_up_sols,
                    min_n_shelf=min_n_shelf,
                    max_n_shelf=max_n_shelf,
                    seed=repair_seed,
                    limit_n_shelf=limit_n_shelf,
                    n_threads=self.config.repair_n_threads,
                    time_limit=self.config.repair_timelimit,
                    agent_num=agent_num,
                )
                if map_np_repaired is None:
                    failed_unrepaired_env = format_env_str(
                        manufacture_env_number2str(map_np_unrepaired))
                    logger.info(
                        f"Repair failed! The layout is:\n{failed_unrepaired_env}"
                    )
                    return None, map_np_unrepaired, None

            # Convert map layout to str format
            map_str_repaired = manufacture_env_number2str(map_np_repaired)

            format_env = format_env_str(map_str_repaired)
            logger.info(f"\nRepaired result:\n{format_env}")

            # Create json string to map layout
            map_json = json.dumps({
                "name": f"sol-seed={sim_seed}",
                "weight": False,
                "n_row": n_row,
                "n_col": n_col,
                "layout": map_str_repaired,
            })

        else:
            NotImplementedError("Other manufacture types not supported yet.")

        return map_json, map_np_unrepaired, map_np_repaired

    def evaluate(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int,
    ):
        """
        Repair map and run simulation

        Args:
            map (np.ndarray): input map in integer format
            parent_map (np.ndarray): parent solution of the map. Will be None if
                                     current sol is the initial population.
            eval_logdir (str): log dir of simulation
            n_evals (int): number of evaluations
            sim_seed (int): random seed for simulation. Should be different for
                            each solution
            repair_seed (int): random seed for repairing. Should be the same as
                               master seed
            w_mode (bool): whether to run with w_mode, which replace 'r' with
                           'w' in generated map layouts, where 'w' is a
                           workstation. Under w_mode, robots will start from
                           endpoints and their tasks will alternate between
                           endpoints and workstations.
            n_endpt (int): number of endpoint around each obstacle
            min_n_shelf (int): min number of shelves
            max_n_shelf (int): max number of shelves
            agentNum (int): number of drives
            map_id (int): id of the current map to be evaluated. The id
                          is only unique to each batch, NOT to the all the
                          solutions. The id can make sure that each simulation
                          gets a different log directory.
        """
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        # We need to construct kwargs manually because some parameters
        # must NOT be passed in in order to use the default values
        # defined on the C++ side.
        # It is very dumb but works.

        kwargs = {
            "map": map_json,
            "output": output,
            "scenario": self.config.scenario,
            "task": self.config.task,
            "agentNum": agentNum,
            "cutoffTime": self.config.cutoffTime,
            "seed": sim_seed,
            "screen": self.config.screen,
            "solver": self.config.solver,
            "id": self.config.id,
            "single_agent_solver": self.config.single_agent_solver,
            "lazyP": self.config.lazyP,
            "simulation_time": self.config.simulation_time,
            "simulation_window": self.config.simulation_window,
            "travel_time_window": self.config.travel_time_window,
            "potential_function": self.config.potential_function,
            "potential_threshold": self.config.potential_threshold,
            "rotation": self.config.rotation,
            "robust": self.config.robust,
            "CAT": self.config.CAT,
            "hold_endpoints": self.config.hold_endpoints,
            "dummy_paths": self.config.dummy_paths,
            "prioritize_start": self.config.prioritize_start,
            "suboptimal_bound": self.config.suboptimal_bound,
            "log": self.config.log,
            "test": self.config.test,
            "force_new_logdir": True,
            "save_result": self.config.save_result,
            "save_solver": self.config.save_solver,
            "save_heuristics_table": self.config.save_heuristics_table,
            "stop_at_traffic_jam": self.config.stop_at_traffic_jam,
            "n_station_types": self.config.n_station_types,
            "station_wait_times": self.config.station_wait_times,
        }

        # For some of the parameters, if they are not privided in the config
        # file, we want to use the default value defined on the C++ side.
        # We are not able to define the default values in python because values
        # such as INT_MAX can be tricky in python but easy in C++.
        planning_window = self.config.planning_window
        if planning_window is not None:
            kwargs["planning_window"] = planning_window
        OverallCutoffTime = self.config.OverallCutoffTime
        if OverallCutoffTime is not None:
            kwargs["OverallCutoffTime"] = OverallCutoffTime

        one_sim_result_jsonstr = warehouse_sim.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        return result_json

    def process_eval_result(
        self,
        curr_result_json: List[dict],
        n_evals: int,
        map_np_unrepaired: np.ndarray,
        map_np_repaired: np.ndarray,
        max_n_shelf: int,
        map_id: int,
    ):
        """
        Process the evaluation result

        Args:
            curr_result_json (List[dict]): result json of all simulations of 1
                map.

        """

        # Deal with failed layout.
        # For now, failure only happens during MILP repair, so it failure
        # happens, all simulation json results would contain
        # {"success": False}.
        if not curr_result_json[0]["success"]:
            logger.info(f"Map ID {map_id} failed.")

            metadata = ManufactureMetadata(
                map_int_unrepaired=map_np_unrepaired, )
            result = ManufactureResult.from_raw(
                manufacture_metadata=metadata,
                opts={
                    "aggregation": self.config.aggregation_type,
                    "measure_names": self.config.measure_names,
                },
            )
            result.failed = True
            return result

        # Collect the results
        keys = curr_result_json[0].keys()
        collected_results = {key: [] for key in keys}
        for result_json in curr_result_json:
            for key in keys:
                collected_results[key].append(result_json[key])

        # Calculate n_shelf and n_endpoint
        # Note: we use the number of tiles in storage area (aka the portion of
        # the layout in the middle) as the totol number of tiles
        tile_ele, tile_cnt = np.unique(map_np_repaired, return_counts=True)
        tile_cnt_dict = dict(zip(tile_ele, tile_cnt))
        n_shelf = tile_cnt_dict[manufacture_obj_types.index("0")] + \
                  tile_cnt_dict[manufacture_obj_types.index("1")] + \
                  tile_cnt_dict[manufacture_obj_types.index("2")]
        n_endpoint = tile_cnt_dict[manufacture_obj_types.index("e")]

        # Get average length of all tasks
        all_task_len_mean = collected_results.get("avg_task_len")
        all_task_len_mean = all_task_len_mean[0]

        logger.info(
            f"Map ID {map_id}: Average length of all possible tasks: {all_task_len_mean}"
        )

        # Calculate number of connected shelf components
        n_shelf_components = calc_num_shelf_components(map_np_repaired)
        logger.info(
            f"Map ID {map_id}: Number of connected shelf components: {n_shelf_components}"
        )

        # Calculate layout entropy
        entropy = calc_layout_entropy(map_np_repaired)
        logger.info(f"Map ID {map_id}: Layout entropy: {entropy}")

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"))
        tile_usage = tile_usage.reshape(n_evals, *map_np_repaired.shape)
        tasks_finished_timestep = [
            np.array(x)
            for x in collected_results.get("tasks_finished_timestep")
        ]

        # Get objective based on type
        objs = None
        throughput = np.array(collected_results.get("throughput"))
        similarity_score = None
        if self.config.obj_type == "throughput":
            objs = throughput
        elif self.config.obj_type == "throughput_plus_n_shelf":
            objs = throughput - \
                (max_n_shelf - n_shelf)**2 * 0.5
        elif self.config.obj_type == "throughput_minus_hamming_dist":
            # Calculate hamming distance
            assert map_np_unrepaired.shape == map_np_repaired.shape

            similarity_score = cal_similarity_score(
                map_np_unrepaired,
                map_np_repaired,
                self.config.station_same_weight,
            )

            # Add throughput
            objs = throughput + self.config.hamming_obj_weight * similarity_score

            ####################### Deprecated #######################
            # hamming_dist = (map_np_unrepaired != map_np_repaired).sum()
            # logger.info(f"Map ID {map_id}: Hamming dist: {hamming_dist}")
            # n_tiles = np.prod(map_np_unrepaired.shape)

            # Normalize hamming dist "regularization" to [0, 1]
            # Essentially we maximize:
            # 1. The throughput
            # 2. The percentage of tiles that are the same in unrepaired and
            #    repaired layouts
            # objs = throughput + self.config.hamming_obj_weight * \
            #         (1 - hamming_dist/n_tiles)
            ####################### Deprecated #######################
            logger.info(
                f"Map ID {map_id}: similarity score: {similarity_score}")
            logger.info(f"Map ID {map_id}: Computed obj: {objs}")
        else:
            return ValueError(
                f"Object type {self.config.obj_type} not supported")

        # Create ManufactureResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = ManufactureMetadata(
            objs=objs,
            throughput=collected_results.get("throughput"),
            map_int_unrepaired=map_np_unrepaired,
            map_int=map_np_repaired,
            # map_int_raw=map_np_unrepaired,
            map_str=manufacture_env_number2str(map_np_repaired),
            n_shelf=n_shelf,
            n_endpoint=n_endpoint,
            tile_usage=tile_usage,
            tile_usage_mean=np.mean(collected_results.get("tile_usage_mean")),
            tile_usage_std=np.mean(collected_results.get("tile_usage_std")),
            num_wait=collected_results.get("num_wait"),
            num_wait_mean=np.mean(collected_results.get("num_wait_mean")),
            num_wait_std=np.mean(collected_results.get("num_wait_std")),
            num_turns=collected_results.get("num_turns"),
            num_turns_mean=np.mean(collected_results.get("num_turns_mean")),
            num_turns_std=np.mean(collected_results.get("num_turns_std")),
            finished_task_len=collected_results.get("finished_task_len"),
            finished_len_mean=np.mean(
                collected_results.get("finished_len_mean")),
            finished_len_std=np.mean(collected_results.get("finished_len_std")),
            all_task_len_mean=all_task_len_mean,
            tasks_finished_timestep=tasks_finished_timestep,
            n_shelf_components=n_shelf_components,
            layout_entropy=entropy,
            cpu_runtime=collected_results.get("cpu_runtime"),
            cpu_runtime_mean=np.mean(collected_results.get("cpu_runtime")),
            similarity_score=similarity_score,
        )
        result = ManufactureResult.from_raw(
            manufacture_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

        return result

    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)


logger = logging.getLogger(__name__)
d = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def cal_similarity_score(
    map_np_unrepaired,
    map_np_repaired,
    station_same_weight,
):
    """Compute the similarity score

    Compute the objective as throughput + weighted_same_sum
    where weighted_same_sum = (map_np_unrepaired != map_np_repaired)
    element-wise multiply a `weight_matrix``.

    The `weight_matrix` is determined as follows: it has the same
    size as the layout. Each entry is
    `self.config.station_same_weight` if the corresponding entry is a
    manufacture station, and 1 other wise.

    Intuitively, we try to reward the manufacture stations that are
    "correctly generated" (i.e. not changed during MILP repair).
    """
    # Get the `same_matrix` where each entry is 1 if the tile is the
    # same in unrepaired and repaired layout, and 0 otherwise
    same_matrix = (map_np_unrepaired == map_np_repaired).astype(float)

    # Get the `weight_matrix``
    weight_matrix = np.ones(map_np_unrepaired.shape)
    weight_matrix[np.where(map_np_unrepaired == manufacture_obj_types.index(
        "0"))] = station_same_weight
    weight_matrix[np.where(map_np_unrepaired == manufacture_obj_types.index(
        "1"))] = station_same_weight
    weight_matrix[np.where(map_np_unrepaired == manufacture_obj_types.index(
        "2"))] = station_same_weight

    # Element-wise multiply and take the sum
    weighted_sum = np.sum(same_matrix * weight_matrix)

    # Normalize to [0, 1]
    n_tiles = np.prod(map_np_unrepaired.shape)
    max_weight_per_tile = max(1, station_same_weight)
    weighted_sum /= max_weight_per_tile * n_tiles
    return weighted_sum


def calc_layout_entropy(map_np_repaired):
    """
    Calculate entropy of the of the layout.

    We first formulate the layout as a tile pattern distribution by following
    Lucas, Simon M. M. and Vanessa Volz. “Tile pattern KL-divergence for
    analysing and evolving game levels.” Proceedings of the Genetic and
    Evolutionary Computation Conference (2019).

    Then we calculate the entropy.
    """
    h, w = map_np_repaired.shape

    # Generate list of patterns (we use 2 x 2)
    storage_obj_types = manufacture_obj_types[:-1]
    tile_patterns = {
        "".join(x): 0
        for x in product(storage_obj_types, repeat=4)
    }

    h, w = map_np_repaired.shape
    # Iterate over 2x2 blocks
    for i in range(h - 1):
        for j in range(w - 1):
            curr_block = map_np_repaired[i:i + 2, j:j + 2]
            curr_pattern = "".join(manufacture_env_number2str(curr_block))
            tile_patterns[curr_pattern] += 1
    pattern_dist = list(tile_patterns.values())

    # Use number of patterns as the base to bound the entropy to [0, 1]
    return entropy(pattern_dist, base=len(pattern_dist))


def BFS_shelf_component(start_loc, env_np, env_visited):
    """
    Find all shelves that are connected to the shelf at start_loc.
    """
    # We must start searching from shelf
    assert env_np[start_loc] == manufacture_obj_types.index("0") or \
           env_np[start_loc] == manufacture_obj_types.index("1") or \
           env_np[start_loc] == manufacture_obj_types.index("2")

    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    block_idxs = [
        manufacture_obj_types.index("e"),
        manufacture_obj_types.index("."),
    ]
    while not q.empty():
        curr = q.get()
        x, y = curr
        env_visited[x, y] = True
        seen.add(curr)
        for dx, dy in d:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))


def calc_num_shelf_components(repaired_env):
    env_visited = np.zeros(repaired_env.shape, dtype=bool)
    n_row, n_col = repaired_env.shape
    n_shelf_components = 0
    for i in range(n_row):
        for j in range(n_col):
            if (repaired_env[i,j] == manufacture_obj_types.index("0") or \
                repaired_env[i,j] == manufacture_obj_types.index("1") or \
                repaired_env[i,j] == manufacture_obj_types.index("2")) and \
                not env_visited[i,j]:
                n_shelf_components += 1
                BFS_shelf_component((i, j), repaired_env, env_visited)
    return n_shelf_components


def single_simulation(seed, agent_num, kwargs, results_dir):
    kwargs["seed"] = int(seed)
    output_dir = os.path.join(results_dir,
                              f"sim-agent_num={agent_num}-seed={seed}")
    kwargs["output"] = output_dir
    kwargs["agentNum"] = agent_num

    result_jsonstr = warehouse_sim.run(**kwargs)
    result_json = json.loads(result_jsonstr)

    throughput = result_json["throughput"]

    # Write result to logdir
    # Load and then dump to format the json
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        f.write(json.dumps(result_json, indent=4))

    # Write kwargs to logdir
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(kwargs, indent=4))

    return throughput


def test_calc_layout_entropy(map_filepath):
    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = manufacture_env_str2number(repaired_env_str)

    layout_entropy = calc_layout_entropy(repaired_env, True)
    print(f"Layout entropy: {layout_entropy}")


def main(
    manufacture_config,
    map_filepath,
    agent_num=10,
    agent_num_step_size=1,
    seed=0,
    n_evals=10,
    n_sim=2,  # Run `inc_agents` `n_sim`` times
    mode="constant",
    n_workers=32,
    reload=None,
):
    """
    For testing purposes. Graph a map and run one simulation.

    Args:
        mode: "constant", "inc_agents", or "inc_timesteps".
              "constant" will run `n_eval` simulations with the same
              `agent_num`.
              "increase" will run `n_eval` simulations with an inc_agents
              number of `agent_num`.
    """
    setup_logging(on_worker=False)

    gin.parse_config_file(manufacture_config)

    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)

    # Create log dir
    map_name = raw_env_json["name"]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + map_name
    log_dir = os.path.join(LOG_DIR, base_log_dir)
    results_dir = os.path.join(log_dir, "results")
    os.mkdir(log_dir)
    os.mkdir(results_dir)

    # Write map file to logdir
    with open(os.path.join(log_dir, "map.json"), "w") as f:
        f.write(json.dumps(raw_env_json, indent=4))

    # Construct kwargs
    kwargs = {
        "map":
        json.dumps(raw_env_json),
        # "output" : log_dir,
        "scenario":
        gin.query_parameter("ManufactureConfig.scenario"),
        "task":
        gin.query_parameter("ManufactureConfig.task"),
        "agentNum":
        agent_num,
        "cutoffTime":
        gin.query_parameter("ManufactureConfig.cutoffTime"),
        # "seed" : seed,
        "screen":
        gin.query_parameter("ManufactureConfig.screen"),
        "solver":
        gin.query_parameter("ManufactureConfig.solver"),
        "id":
        gin.query_parameter("ManufactureConfig.id"),
        "single_agent_solver":
        gin.query_parameter("ManufactureConfig.single_agent_solver"),
        "lazyP":
        gin.query_parameter("ManufactureConfig.lazyP"),
        "simulation_time":
        gin.query_parameter("ManufactureConfig.simulation_time"),
        "simulation_window":
        gin.query_parameter("ManufactureConfig.simulation_window"),
        "travel_time_window":
        gin.query_parameter("ManufactureConfig.travel_time_window"),
        "potential_function":
        gin.query_parameter("ManufactureConfig.potential_function"),
        "potential_threshold":
        gin.query_parameter("ManufactureConfig.potential_threshold"),
        "rotation":
        gin.query_parameter("ManufactureConfig.rotation"),
        "robust":
        gin.query_parameter("ManufactureConfig.robust"),
        "CAT":
        gin.query_parameter("ManufactureConfig.CAT"),
        "hold_endpoints":
        gin.query_parameter("ManufactureConfig.hold_endpoints"),
        "dummy_paths":
        gin.query_parameter("ManufactureConfig.dummy_paths"),
        "prioritize_start":
        gin.query_parameter("ManufactureConfig.prioritize_start"),
        "suboptimal_bound":
        gin.query_parameter("ManufactureConfig.suboptimal_bound"),
        "log":
        gin.query_parameter("ManufactureConfig.log"),
        "test":
        gin.query_parameter("ManufactureConfig.test"),
        "force_new_logdir":
        False,
        "save_result":
        gin.query_parameter("ManufactureConfig.save_result"),
        "save_solver":
        gin.query_parameter("ManufactureConfig.save_solver"),
        "save_heuristics_table":
        gin.query_parameter("ManufactureConfig.save_heuristics_table"),
        "stop_at_traffic_jam":
        gin.query_parameter("ManufactureConfig.stop_at_traffic_jam"),
        "n_station_types":
        gin.query_parameter("ManufactureConfig.n_station_types"),
        "station_wait_times":
        gin.query_parameter("ManufactureConfig.station_wait_times"),
    }

    # For some of the parameters, if they are not privided in the config
    # file, we want to use the default value defined on the C++ side.
    # We are not able to define the default values in python because values
    # such as INT_MAX can be tricky in python but easy in C++.
    try:
        planning_window = gin.query_parameter(
            "ManufactureConfig.planning_window")
        if planning_window is not None:
            kwargs["planning_window"] = planning_window

        OverallCutoffTime = gin.query_parameter(
            "ManufactureConfig.OverallCutoffTime")
        if OverallCutoffTime is not None:
            kwargs["OverallCutoffTime"] = OverallCutoffTime

    except ValueError:
        pass

    have_run = set()
    all_results_dir = os.path.join(reload, "results")
    if reload is not None and reload != "":
        for result_dir in os.listdir(all_results_dir):
            result_dir_full = os.path.join(all_results_dir, result_dir)
            if os.path.exists(os.path.join(result_dir_full, "result.json")) and\
               os.path.exists(os.path.join(result_dir_full, "config.json")):
                curr_configs = result_dir.split("-")
                curr_agent_num = int(curr_configs[1].split("=")[1])
                curr_seed = int(curr_configs[2].split("=")[1])
                have_run.add((curr_agent_num, curr_seed))
            else:
                breakpoint()
                shutil.rmtree(result_dir_full)

    pool = multiprocessing.Pool(n_workers)
    if mode == "inc_agents":
        seeds = []
        agent_nums = []
        agent_num_range = range(0, n_evals, agent_num_step_size)
        actual_n_evals = len(agent_num_range)
        for i in range(n_sim):
            for j in agent_num_range:
                curr_seed = seed + i
                curr_agent_num = agent_num + j
                if (curr_agent_num, curr_seed) in have_run:
                    continue
                seeds.append(curr_seed)
                agent_nums.append(curr_agent_num)
        throughputs = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, actual_n_evals * n_sim - len(have_run)),
                repeat(results_dir, actual_n_evals * n_sim - len(have_run))),
        )
    elif mode == "constant":
        agent_nums = [agent_num for _ in range(n_evals)]
        seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)

        throughputs = pool.starmap(
            single_simulation,
            zip(seeds, agent_nums, repeat(kwargs, n_evals),
                repeat(results_dir, n_evals)),
        )

    avg_obj = np.mean(throughputs)
    max_obj = np.max(throughputs)
    min_obj = np.min(throughputs)

    n_evals = actual_n_evals if mode == "inc_agents" else n_evals

    print(f"Average throughput over {n_evals} simulations: {avg_obj}")
    print(f"Max throughput over {n_evals} simulations: {max_obj}")
    print(f"Min throughput over {n_evals} simulations: {min_obj}")


if __name__ == "__main__":
    fire.Fire(main)
