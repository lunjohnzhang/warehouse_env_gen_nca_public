import gin
import time
import fire
import torch
import numpy as np

# Including this makes gin config work because main imports (pretty much)
# everything.
import env_search.main  # pylint: disable = unused-import

from env_search.device import DEVICE
from env_search.utils.logging import setup_logging
from env_search.utils import (read_in_maze_map, format_env_str, n_params)
from env_search.maze.generator.nca_generator import MazeNCA
from env_search.maze.level import MazeLevel


def generate_with_time(mazeNCA, seed_map_int):
    start_time = time.time()
    out, _ = mazeNCA.generate(
        torch.tensor(seed_map_int[np.newaxis, :, :], device=DEVICE),
        n_iter=10,
    )
    time_elapsed = time.time() - start_time
    print(format_env_str(MazeLevel.number_to_str(out.squeeze().cpu().numpy())))
    print("Time taken: ", time_elapsed)


def test_nca_generator(maze_config: str, seed_env_path: str):
    setup_logging(on_worker=False)
    gin.clear_config()
    gin.parse_config_file(maze_config)

    # Read in seed map
    seed_map_str, _ = read_in_maze_map(seed_env_path)
    seed_map_int = MazeLevel.str_to_number(seed_map_str)

    mazeNCA = MazeNCA().to(DEVICE)

    num_params = n_params(mazeNCA)
    print("Number of params: ", num_params)
    generate_with_time(mazeNCA, seed_map_int)

    # Set parameter and try again
    new_params = np.random.rand(num_params)
    mazeNCA.set_params(new_params)
    generate_with_time(mazeNCA, seed_map_int)


if __name__ == '__main__':
    fire.Fire(test_nca_generator)