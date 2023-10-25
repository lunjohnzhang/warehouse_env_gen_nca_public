import gin
import torch
import numpy as np

from torch import nn
from env_search.warehouse.generator.nca_generator import WarehouseNCA


@gin.configurable
class ManufactureNCA(WarehouseNCA):
    """NCA module based on the architecture described in the paper
    "Illuminating Diverse Neural Cellular Automata for Level Generation" by
    Sam Earle, Justin Snider, Matthew C. Fontaine, Stefanos Nikolaidis, and
    Julian Togelius, in Proceedings of the Genetic and Evolutionary Computation
    Conference, 2022.

    Args:
        i_size (int): size of input image
        nc (int): total number of objects in the environment
        n_aux_chan (int): number of auxiliary channels
    """
    def __init__(
        self,
        nc: int = gin.REQUIRED,
        n_aux_chan: int = gin.REQUIRED,
        kernel_size: int = gin.REQUIRED,
        n_hid_chan: int = gin.REQUIRED,
    ):
        super().__init__(
            nc=nc,
            n_aux_chan=n_aux_chan,
            kernel_size=kernel_size,
            n_hid_chan=n_hid_chan,
        )
