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
from env_search.manufacture.generator.nca_generator import ManufactureNCA
from env_search.utils import (read_in_manufacture_map,
                              manufacture_env_str2number,
                              manufacture_env_number2str, format_env_str,
                              n_params)


def generate_with_time(manufactureNCA, seed_map_int):
    start_time = time.time()
    out, _ = manufactureNCA.generate(
        torch.tensor(seed_map_int[np.newaxis, :, :], device=DEVICE),
        n_iter=100,
    )
    time_elapsed = time.time() - start_time
    print(
        format_env_str(manufacture_env_number2str(out.squeeze().cpu().numpy())))
    print("Time taken: ", time_elapsed)


def test_nca_generator(manufacture_config: str, seed_env_path: str):
    setup_logging(on_worker=False)
    gin.clear_config()
    gin.parse_config_file(manufacture_config)

    # Read in seed map
    seed_map_str, _ = read_in_manufacture_map(seed_env_path)
    seed_map_int = manufacture_env_str2number(seed_map_str)

    manufactureNCA = ManufactureNCA().to(DEVICE)

    num_params = n_params(manufactureNCA)
    print("Number of params: ", num_params)
    generate_with_time(manufactureNCA, seed_map_int)

    # Set parameter and try again
    new_params = np.random.rand(num_params)
    manufactureNCA.set_params(new_params)
    generate_with_time(manufactureNCA, seed_map_int)


if __name__ == '__main__':
    fire.Fire(test_nca_generator)