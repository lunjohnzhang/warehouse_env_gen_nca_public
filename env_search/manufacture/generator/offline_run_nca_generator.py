import os
import gin
import time
import fire
import torch
import numpy as np
import pandas as pd
import shutil
import sys

np.set_printoptions(threshold=sys.maxsize)

# Including this makes gin config work because main imports (pretty much)
# everything.
import env_search.main  # pylint: disable = unused-import

from env_search.analysis.utils import (load_experiment, load_metrics,
                                       load_archive_gen,
                                       load_archive_from_reload)
from env_search.analysis.visualize_env import visualize_manufacture, create_movie
from env_search.manufacture.module import cal_similarity_score
from env_search.manufacture.milp_repair import repair_env
from env_search.device import DEVICE
from env_search.utils.logging import setup_logging
from env_search.manufacture.generator.nca_generator import ManufactureNCA
from env_search.utils import (manufacture_obj_types,
                              manufacture_env_number2str,
                              manufacture_env_str2number, format_env_str,
                              read_in_manufacture_map, n_params,
                              write_map_str_to_json)


def generate_nca_evo_process(all_sols, save_dir):
    for i, sol in enumerate(all_sols):
        # print(format_env_str(manufacture_env_number2str(sol)))
        visualize_manufacture(
            sol,
            filenames=[f"gen_{i:04d}.png"],
            store_dir=save_dir,
            dpi=200,
            figsize=(8, 8),
        )


def generate_with_time(
    manufactureNCA,
    seed_map_int,
    nca_process_dir,
    save=True,
    nca_iter=200,
):
    start_time = time.time()
    out, all_sols = manufactureNCA.generate(
        torch.tensor(seed_map_int[np.newaxis, :, :], device=DEVICE),
        n_iter=nca_iter,
        save=save,
    )
    nca_runtime = time.time() - start_time
    generate_nca_evo_process(all_sols, nca_process_dir)
    create_movie(nca_process_dir, "nca_process")

    out = out.squeeze().cpu().numpy()
    # print(format_env_str(manufacture_env_number2str(out)))
    print("NCA taken: ", nca_runtime)
    return out, nca_runtime


def offline_run_nca_generator(
    logdir: str,
    seed_env_path: str,
    gen: int = None,
    mode: str = "best",  # one of ["best", "idx"]
    query: "array-like" = None,  # type: ignore
    nca_only: bool = False,
    nca_iter: int = 200,
    sim_score_with: str = None,
):
    """
    Load trained NCA and run it once with the specified seed.

    Args:
        logdir: logdir of the experiment
        seed_env_path: path to the NCA seed
        mode: method to choose solution from archive.
            1. "best": use optimal NCA from the archive
            2. "query": use NCA of specific index from the archive
    """
    logdir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    archive = load_archive_gen(logdir, gen)

    optimize_archive, result_archive = load_archive_from_reload(
        logdir, is_cma_mae=True)

    # df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))
    df = result_archive.as_pandas(include_solutions=True,
                                  include_metadata=True)

    # Obtain global optima from the archive
    if mode == "best":
        nca_idx = df["objective"].idxmax()

    elif mode == "idx":
        query = tuple(query)
        nca_idx, = np.where(
            df.index_batch() == archive.grid_to_int_index([query])[0])
        if len(nca_idx) == 0:
            raise ValueError(f"Index {query} not available")
        nca_idx = nca_idx[0]

    map_str_repaired = df.iloc[nca_idx]["metadata"]["manufacture_metadata"][
        "map_str"]
    if mode == "best":
        print("Global optima: ")
    elif mode == "idx":
        print("Env of index", query)
    throughput = df.iloc[nca_idx]["metadata"]["manufacture_metadata"][
        "throughput"]
    objs = df.iloc[nca_idx]["metadata"]["manufacture_metadata"]["objs"]

    # Calculate percentage of same tiles of repaired and unrepaired
    map_np_repaired = manufacture_env_str2number(map_str_repaired)
    map_np_unrepaired = df.iloc[nca_idx]["metadata"]["manufacture_metadata"][
        "map_int_unrepaired"]
    map_str_unrepaired = manufacture_env_number2str(map_np_unrepaired)
    n_tiles = np.prod(map_np_unrepaired.shape)
    hamming_dist = (map_np_unrepaired != map_np_repaired).sum()
    percent_same_tile = (1 - hamming_dist / n_tiles)
    sim_score_search = cal_similarity_score(map_np_unrepaired, map_np_repaired,
                                            5)

    # Print stats from search
    print(f"throughputs: {throughput}")
    print(f"objectives: {objs}")
    print(f"Hamming dist: {hamming_dist}")
    print(f"Percent same tiles: {percent_same_tile}")
    print(f"Weighted same score: {sim_score_search}")
    print("Unrepaired from search: ")
    print("\n".join(map_str_unrepaired))
    print()
    print("Repaired from search: ")
    print("\n".join(map_str_repaired))
    print()

    # breakpoint()

    # Read in seed map
    seed_map_str, _ = read_in_manufacture_map(seed_env_path)
    seed_map_int = manufacture_env_str2number(seed_map_str)

    # Infer the size
    env_w, env_h = seed_map_int.shape
    env_size = ""
    if env_w * env_h == 33 * 36:
        env_size = "large"
    elif env_w * env_h == 101 * 102:
        env_size = "xxlarge"
    else:
        env_size = f"{env_w}x{env_h}"

    # Create sub-logdir for nca process inside the given logidr
    nca_process_dir = logdir.dir(f"nca_process_{env_size}_iter={nca_iter}")
    if os.path.isdir(nca_process_dir):
        shutil.rmtree(nca_process_dir, ignore_errors=True)
    os.mkdir(nca_process_dir)

    write_map_str_to_json(
        os.path.join(nca_process_dir, "unrepaired_from_search.json"),
        map_str_unrepaired,
        "unrepaired_from_search",
        "manufacture",
        sim_score=sim_score_search,
    )

    write_map_str_to_json(
        os.path.join(nca_process_dir, "repaired_from_search.json"),
        map_str_repaired,
        "repaired_from_search",
        "manufacture",
        sim_score=sim_score_search,
    )

    nca_params = np.array(df.filter(regex=("solution_*")).iloc[nca_idx])

    manufactureNCA = ManufactureNCA().to(DEVICE)

    num_params = n_params(manufactureNCA)
    print("Number of params: ", num_params)

    # Set parameter and run
    manufactureNCA.set_params(nca_params)
    unrepaired_layout, nca_runtime = generate_with_time(
        manufactureNCA,
        seed_map_int,
        nca_process_dir,
        nca_iter=nca_iter,
    )
    sim_score_gen = None
    if sim_score_with is not None and sim_score_with != "":
        # Read in layout from sim_score_with
        sim_score_with_map_str, _ = read_in_manufacture_map(sim_score_with)
        sim_score_with_map_np = manufacture_env_str2number(
            sim_score_with_map_str)
        sim_score_gen = cal_similarity_score(unrepaired_layout,
                                             sim_score_with_map_np, 5)

    map_str_unrepaired_nca = manufacture_env_number2str(unrepaired_layout)
    write_map_str_to_json(
        os.path.join(nca_process_dir, "unrepaired_nca_gen.json"),
        map_str_unrepaired_nca,
        "unrepaired_nca_gen",
        "manufacture",
        sim_score=sim_score_gen,
    )

    if nca_only:
        return

    print("========= Start repairing NCA generated layout =========")
    print(f"Unrepaired fron NCA generation with iter={nca_iter}: ")
    print(format_env_str(map_str_unrepaired_nca))
    print()

    # MILP repair
    repair_time_limit = -1
    repair_n_threads = -1
    if env_w * env_h <= 33 * 36:
        repair_time_limit = 60
        repair_n_threads = 1
    else:
        repair_time_limit = 7200
        repair_n_threads = 28

    start_time = time.time()

    print(
        f"Repairing with {repair_n_threads} threads and time limit {repair_time_limit}"
    )
    hamming_repaired_env = repair_env(
        unrepaired_layout,
        add_movement=False,
        min_n_shelf=0,
        max_n_shelf=np.prod(seed_map_int.shape, dtype=int),
        seed=0 if env_size == "xxlarge" else 42,
        # warm_envs_np=warm_up_sols,
        limit_n_shelf=True,
        n_threads=repair_n_threads,
        time_limit=repair_time_limit,
    )
    milp_runtime = time.time() - start_time

    map_str_repaired_nca = manufacture_env_number2str(hamming_repaired_env)
    write_map_str_to_json(
        os.path.join(nca_process_dir, "repaired_nca_gen.json"),
        map_str_repaired_nca,
        "repaired_nca_gen",
        "manufacture",
        nca_runtime=nca_runtime,
        milp_runtime=milp_runtime,
        nca_milp_runtime=nca_runtime + milp_runtime,
    )

    print("Repaired: ")
    print(format_env_str(map_str_repaired_nca))

    print("Repair taken: ", milp_runtime)


if __name__ == '__main__':
    fire.Fire(offline_run_nca_generator)