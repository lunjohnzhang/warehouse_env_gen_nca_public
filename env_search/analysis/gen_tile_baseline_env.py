import fire
import time
import numpy as np

from logdir import LogDir
from env_search.utils import (
    read_in_kiva_map,
    read_in_manufacture_map,
    read_in_maze_map,
    kiva_env_str2number,
    kiva_env_number2str,
    manufacture_env_str2number,
    manufacture_env_number2str,
    maze_env_str2number,
    write_map_str_to_json,
    maze_env_number2str,
    format_env_str,
    KIVA_WORKSTATION_BLOCK_WIDTH,
)
from env_search.warehouse.milp_repair import repair_env as kiva_repair
from env_search.manufacture.milp_repair import repair_env as manufacture_repair
from env_search.warehouse.module import add_non_storage_area


def repeat_and_crop(small_env_file, N, M, domain):
    if domain == "kiva":
        small_env, env_name = read_in_kiva_map(small_env_file)
        small_env = kiva_env_str2number(small_env)
        # Remove non-storage area
        width = small_env.shape[1]
        small_env = small_env[:, KIVA_WORKSTATION_BLOCK_WIDTH:width -
                              KIVA_WORKSTATION_BLOCK_WIDTH]
    elif domain == "manufacture":
        small_env, env_name = read_in_manufacture_map(small_env_file)
        small_env = manufacture_env_str2number(small_env)
    elif domain == "maze":
        small_env, env_name = read_in_maze_map(small_env_file)
        small_env = maze_env_str2number(small_env)

    # Calculate how many times to repeat the array along each axis
    # equivalent to math.ceil(N / array.shape[0])
    tiles_row = -(-N // small_env.shape[0])
    # equivalent to math.ceil(M / array.shape[1])
    tiles_col = -(-M // small_env.shape[1])

    # Tile the array
    large_env_unrepaired = np.tile(small_env, (tiles_row, tiles_col))

    # Crop the array to the desired size
    large_env_unrepaired = large_env_unrepaired[:N, :M]

    if domain == "kiva":
        large_env_unrepaired, n_row_comp, n_col_comp = add_non_storage_area(
            large_env_unrepaired,
            w_mode=True,
        )
        large_env_unrepaired_str = kiva_env_number2str(large_env_unrepaired)

    elif domain == "manufacture":
        large_env_unrepaired_str = manufacture_env_number2str(
            large_env_unrepaired)
    elif domain == "maze":
        large_env_unrepaired_str = maze_env_number2str(large_env_unrepaired)

    logdir = LogDir(
        f"{N}x{M}_{domain}_baseline_env_gen",
        rootdir="./logs",
        uuid=True,
    )
    unrepaired_env_name = f"{domain}_{N}x{M}_tile_baseline_unrepaired"
    write_map_str_to_json(
        logdir.file(f"{unrepaired_env_name}.json"),
        large_env_unrepaired_str,
        unrepaired_env_name,
        domain,
    )
    print("Unrepaired: ")
    print(format_env_str(large_env_unrepaired_str))
    print()

    # Repair if necessary
    # Maze does not need to repair
    start_time = time.time()
    if domain == "maze":
        return
    elif domain == "kiva":
        large_env_repaired = kiva_repair(
            large_env_unrepaired,
            agent_num=1000,
            time_limit=3600,
            add_movement=False,
            w_mode=True,
            max_n_shelf=2500,
            min_n_shelf=2500,
            n_threads=28,
        )
        large_env_repaired_str = kiva_env_number2str(large_env_repaired)
    elif domain == "manufacture":
        large_env_repaired = manufacture_repair(
            large_env_unrepaired,
            agent_num=1000,
            time_limit=3600,
            add_movement=False,
            max_n_shelf=int(N * M),
            min_n_shelf=0,
            n_threads=28,
        )
        large_env_repaired_str = manufacture_env_number2str(large_env_repaired)
    milp_runtime = time.time() - start_time

    repaired_env_name = f"{domain}_{N}x{M}_tile_baseline_repaired"
    write_map_str_to_json(
        logdir.file(f"{repaired_env_name}.json"),
        large_env_repaired_str,
        repaired_env_name,
        domain,
        milp_runtime=milp_runtime,
    )

    print("Repaired: ")
    print(format_env_str(large_env_repaired_str))
    print("Repair taken: ", milp_runtime)


if __name__ == "__main__":
    fire.Fire(repeat_and_crop)