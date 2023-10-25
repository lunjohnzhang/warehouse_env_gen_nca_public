import os
import gin
import time
import fire
import torch
import numpy as np
import pandas as pd
import shutil

# Including this makes gin config work because main imports (pretty much)
# everything.
import env_search.main  # pylint: disable = unused-import

from env_search.warehouse.module import add_non_storage_area, cal_similarity_score
from env_search.analysis.utils import (load_experiment, load_metrics,
                                       load_archive_gen)
from env_search.analysis.visualize_env import visualize_kiva, create_movie
from env_search.warehouse.milp_repair import repair_env
from env_search.device import DEVICE
from env_search.utils.logging import setup_logging
from env_search.warehouse.generator.nca_generator import WarehouseNCA
from env_search.utils import (kiva_obj_types, KIVA_ROBOT_BLOCK_WIDTH,
                              KIVA_WORKSTATION_BLOCK_WIDTH, MIN_SCORE,
                              KIVA_ROBOT_BLOCK_HEIGHT, kiva_env_number2str,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles, n_params,
                              write_map_str_to_json)


def generate_nca_evo_process(all_sols, save_dir):
    os.mkdir(save_dir)
    for i, sol in enumerate(all_sols):
        # print(format_env_str(kiva_env_number2str(sol)))
        visualize_kiva(
            sol,
            filenames=[f"gen_{i:04d}.png"],
            store_dir=save_dir,
            dpi=200,
            figsize=(8, 8),
        )


def generate_with_time(
    warehouseNCA,
    seed_map_int,
    nca_process_dir,
    save=True,
    nca_iter=200,
):
    start_time = time.time()
    out, all_sols = warehouseNCA.generate(
        torch.tensor(seed_map_int[np.newaxis, :, :], device=DEVICE),
        n_iter=nca_iter,
        save=save,
    )
    nca_runtime = time.time() - start_time

    generate_nca_evo_process(all_sols, nca_process_dir)
    create_movie(nca_process_dir, "nca_process")

    out = out.squeeze().cpu().numpy()
    # print(format_env_str(kiva_env_number2str(out)))
    print("NCA taken: ", nca_runtime)
    return out, nca_runtime


def offline_run_nca_generator(
    logdir: str,
    seed_env_path: str,
    gen: int = None,
    mode: str = "optimal",
    nca_only: bool = False,
    nca_iter: int = 200,
    sim_score_with: str = None,
):
    """
    Load trained NCA and run it once with the specified seed.

    Args:
        logdir: logdir of the experiment
        seed_env_path: path to the NCA seed
        mode: 1. "optimal": use optimal NCA from the archive
        nca_only: If true, would only generate unreparied env
        nca_iter: Number of NCA iterations to run
        sim_score_with: map filepath which the unrepaired env to compute sim
            score with.

    """
    logdir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    archive = load_archive_gen(logdir, gen)
    df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))

    # Obtain global optima from the archive
    global_opt = df["objective"].idxmax()
    global_opt_env = df.iloc[global_opt]["metadata"]["warehouse_metadata"][
        "map_str"]
    # global_opt_env_str = kiva_env_str2number(global_opt_env)
    # map_np_unrepaired = df.iloc[global_opt]["metadata"]["warehouse_metadata"][
    #     "map_int_unrepaired"]
    # sim_score_search = cal_similarity_score(map_np_unrepaired, global_opt_env_str)

    # Print stats from search
    print("Global optima: ")
    print("\n".join(global_opt_env))
    print()
    global_opt_nca_params = np.array(
        df.filter(regex=("solution_*")).iloc[global_opt])

    # Read in seed map
    seed_map_str, _ = read_in_kiva_map(seed_env_path)
    seed_map_int = kiva_env_str2number(seed_map_str)

    warehouseNCA = WarehouseNCA().to(DEVICE)

    num_params = n_params(warehouseNCA)
    print("Number of params: ", num_params)

    env_w, env_h = seed_map_int.shape
    env_size = ""
    n_shelf = -1
    repair_time_limit = -1
    repair_n_threads = -1
    if env_w * env_h == 33 * 32:
        env_size = "large"
        n_shelf = 240
        repair_time_limit = 60
        repair_n_threads = 1
    elif env_w * env_h == 17 * 12:
        env_size = "medium"
        n_shelf = 40
        repair_time_limit = 60
        repair_n_threads = 1
    elif env_w * env_h == 101 * 98:
        env_size = "xxlarge"
        n_shelf = 2500
        repair_time_limit = 3600
        repair_n_threads = 28
    else:
        env_size = f"{env_w}x{env_h}"
        repair_time_limit = 3600
        repair_n_threads = 28
        if env_w * env_h == 45 * 43:
            n_shelf = 440
        elif env_w * env_h == 57 * 54:
            n_shelf = 700
        elif env_w * env_h == 69 * 65:
            n_shelf = 1020
        elif env_w * env_h == 81 * 76:
            n_shelf = 1400
        elif env_w * env_h == 93 * 87:
            n_shelf = 1840

    # Set parameter and run
    nca_process_dir = logdir.dir(f"nca_process_{env_size}_iter={nca_iter}")
    if os.path.isdir(nca_process_dir):
        shutil.rmtree(nca_process_dir, ignore_errors=True)
    warehouseNCA.set_params(global_opt_nca_params)
    storage_area, nca_runtime = generate_with_time(
        warehouseNCA,
        seed_map_int,
        nca_process_dir,
        nca_iter=nca_iter,
    )

    # Add workstations/home locations
    w_mode = gin.query_parameter("WarehouseManager.w_mode")
    unrepaired_layout, *_ = add_non_storage_area(storage_area, w_mode)
    print("Unrepaired: ")
    print(format_env_str(kiva_env_number2str(unrepaired_layout)))
    print()

    sim_score_gen = None
    if sim_score_with is not None and sim_score_with != "":
        # Read in layout from sim_score_with
        sim_score_with_map_str, _ = read_in_kiva_map(sim_score_with)
        sim_score_with_map_np = kiva_env_str2number(sim_score_with_map_str)
        _, sim_score_gen = cal_similarity_score(
            unrepaired_layout,
            sim_score_with_map_np,
        )

    write_map_str_to_json(
        os.path.join(nca_process_dir, "unrepaired_nca_gen.json"),
        kiva_env_number2str(unrepaired_layout),
        "unrepaired_nca_gen",
        "kiva",
        nca_runtime=nca_runtime,
        sim_score=sim_score_gen,
    )

    if nca_only:
        return

    # MILP repair
    start_time = time.time()

    hamming_repaired_env = repair_env(
        unrepaired_layout,
        agent_num=1000 if env_size == "xxlarge" else 200,
        add_movement=False,
        min_n_shelf=n_shelf,
        max_n_shelf=n_shelf,
        seed=0 if env_size == "xxlarge" else 42,
        w_mode=w_mode,
        # warm_envs_np=warm_up_sols,
        limit_n_shelf=True,
        n_threads=repair_n_threads,
        time_limit=repair_time_limit,
    )
    milp_runtime = time.time() - start_time

    map_str_repaired_str = kiva_env_number2str(hamming_repaired_env)
    write_map_str_to_json(
        os.path.join(nca_process_dir, "repaired_nca_gen.json"),
        map_str_repaired_str,
        "repaired_nca_gen",
        "kiva",
        nca_runtime=nca_runtime,
        milp_runtime=milp_runtime,
        nca_milp_runtime=nca_runtime + milp_runtime,
    )

    print("Repaired: ")
    print(format_env_str(map_str_repaired_str))

    print("Repair taken: ", milp_runtime)


if __name__ == '__main__':
    fire.Fire(offline_run_nca_generator)