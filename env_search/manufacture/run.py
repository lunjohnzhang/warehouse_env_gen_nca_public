import time
import random
import logging
import pathlib
import json

import traceback
import numpy as np

from typing import List
from env_search.manufacture.manufacture_result import ManufactureResult
from env_search.utils.worker_state import get_manufacture_module
from env_search.manufacture.module import ManufactureModule
from env_search.utils import format_env_str

logger = logging.getLogger(__name__)


def repair_manufacture(
    map: np.ndarray,
    parent_map: np.ndarray,
    sim_seed: int,
    repair_seed: int,
    min_n_shelf: int,
    max_n_shelf: int,
    agent_num: int,
):
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("repair manufacture with seed %d", repair_seed)
    manufacture_module = get_manufacture_module()

    try:
        (
            map_json,
            map_np_unrepaired,
            map_np_repaired,
        ) = manufacture_module.repair(
            map=map,
            parent_map=parent_map,
            sim_seed=sim_seed,
            repair_seed=repair_seed,
            min_n_shelf=min_n_shelf,
            max_n_shelf=max_n_shelf,
            agent_num=agent_num,
        )
    except TimeoutError as e:
        logger.warning(f"repair failed")
        logger.info(f"The map was {map}")
        (
            map_json,
            map_np_unrepaired,
            map_np_repaired,
        ) = [None] * 3

    logger.info("repair_manufacture done after %f sec", time.time() - start)

    return map_json, map_np_unrepaired, map_np_repaired


def run_manufacture(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
) -> ManufactureResult:
    """
    Grabs the manufacture module and evaluates map.

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
        n_endpt (int): number of endpoint around each obstacle
        agentNum (int): number of drives
        map_id (int): id of the current map to be evaluated. The id
                      is only unique to each batch, NOT to the all the
                      solutions. The id can make sure that each simulation
                      gets a different log directory.
    """

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run manufacture with seed %d", sim_seed)
    manufacture_module = get_manufacture_module()
    map_run = format_env_str(json.loads(map_json)["layout"])
    logger.info("Map:")
    print(map_run)
    print()

    try:
        result = manufacture_module.evaluate(
            map_json=map_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            agentNum=agentNum,
            map_id=map_id,
            eval_id=eval_id,
        )
        result["success"] = True
    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {"success": False}

    logger.info("run_manufacture done after %f sec", time.time() - start)

    return result


def process_manufacture_eval_result(
    curr_result_json: List[dict],
    n_evals: int,
    map_np_unrepaired,
    map_np_repaired,
    max_n_shelf: int,
    map_id: int,
):
    start = time.time()

    manufacture_module = get_manufacture_module()

    results = manufacture_module.process_eval_result(
        curr_result_json=curr_result_json,
        n_evals=n_evals,
        map_np_unrepaired=map_np_unrepaired,
        map_np_repaired=map_np_repaired,
        max_n_shelf=max_n_shelf,
        map_id=map_id,
    )
    logger.info("process_manufacture_eval_result done after %f sec",
                time.time() - start)

    return results