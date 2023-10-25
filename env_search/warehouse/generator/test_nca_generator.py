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
from env_search.warehouse.generator.nca_generator import WarehouseNCA
from env_search.utils import (read_in_kiva_map, kiva_env_str2number,
                              kiva_env_number2str, format_env_str, n_params)


def generate_with_time(warehouseNCA, seed_map_int):
    start_time = time.time()
    out, _ = warehouseNCA.generate(
        torch.tensor(seed_map_int[np.newaxis, :, :], device=DEVICE),
        n_iter=10,
    )
    time_elapsed = time.time() - start_time
    print(format_env_str(kiva_env_number2str(out.squeeze().cpu().numpy())))
    print("Time taken: ", time_elapsed)


def test_nca_generator(warehouse_config: str, seed_env_path: str):
    setup_logging(on_worker=False)
    gin.clear_config()
    gin.parse_config_file(warehouse_config)

    # Read in seed map
    seed_map_str, _ = read_in_kiva_map(seed_env_path)
    seed_map_int = kiva_env_str2number(seed_map_str)

    warehouseNCA = WarehouseNCA().to(DEVICE)

    num_params = n_params(warehouseNCA)
    print("Number of params: ", num_params)
    generate_with_time(warehouseNCA, seed_map_int)

    # Set parameter and try again
    new_params = np.random.rand(num_params)
    warehouseNCA.set_params(new_params)
    generate_with_time(warehouseNCA, seed_map_int)


if __name__ == '__main__':
    fire.Fire(test_nca_generator)