from typing import Tuple, List

import gin
import torch
from torch import nn
from torchvision.models import resnet

from env_search.utils.network import int_preprocess
from env_search.utils import manufacture_obj_types

from env_search.warehouse.emulation_model.networks import (
    WarehouseConvolutional, WarehouseAugResnetOccupancy,
    WarehouseAugResnetRepairedMapAndOccupancy)


@gin.configurable
class ManufactureConvolutional(WarehouseConvolutional):
    """Model based on discriminator described in V. Volz, J. Schrum, J. Liu, S.
    M. Lucas, A. Smith, and S. Risi, “Evolving mario levels in the latent
    space of a deep convolutional generative adversarial network,” in
    Proceedings of the Genetic and Evolutionary Computation Conference, 2018.

    Args:
        i_size (int): size of input image
        nc (int): total number of objects in the environment
        ndf (int): number of output channels of initial conv2d layer
        n_extra_layers (int): number of extra layers with out_channels = ndf to
            add
        head_dimensions (List): List of dimensions of the objective and measure
            heads
    """

    def __init__(
        self,
        i_size: int = gin.REQUIRED,
        nc: int = gin.REQUIRED,
        ndf: int = gin.REQUIRED,
        n_extra_layers: int = gin.REQUIRED,
        head_dimensions: List = gin.REQUIRED,
    ):
        super().__init__(
            i_size=i_size,
            nc=nc,
            ndf=ndf,
            n_extra_layers=n_extra_layers,
            head_dimensions=head_dimensions,
        )

    def predict_objs_and_measures(
            self,
            lvls: torch.Tensor,
            aug_lvls: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """Predicts objectives and measures when given int levels.

        Args:
            lvls: (n, lvl_height, lvl_width) tensor of int levels.
            aug_lvls: (n, nc_aug, lvl_height, lvl_width) tensor of predicted aug
                data. This data is concatenated with the onehot version of the
                level as additional channels to the network. Set to None to not
                use aug data. (default: None)
        Returns:
            predicted objectives and predicted measures
        """
        inputs = int_preprocess(lvls, self.i_size, self.nc,
                                manufacture_obj_types.index("."))
        if aug_lvls is not None:
            inputs[:, -aug_lvls.shape[1]:, ...] = aug_lvls
        return self(inputs)


@gin.configurable
class ManufactureAugResnetOccupancy(WarehouseAugResnetOccupancy):
    """Resnet for predicting the agent cell occupancy (aka tile usage) on
    manufacture map.

    Args:
        i_size (int): size of input image.
        nc (int): number of input channels.
        ndf (int): number of output channels of conv2d layer.
        n_res_layers (int): number of residual layers (2x conv per residual
            layer).
        n_out (int): number of outputs.
    """

    def __init__(
        self,
        i_size: int = gin.REQUIRED,
        nc: int = gin.REQUIRED,
        ndf: int = gin.REQUIRED,
        n_res_layers: int = gin.REQUIRED,
        n_out: int = 1,
    ):
        super().__init__(
            i_size=i_size,
            nc=nc,
            ndf=ndf,
            n_res_layers=n_res_layers,
            n_out=n_out,
        )

    def int_to_no_crop(self, lvls: torch.Tensor) -> torch.Tensor:
        inputs = int_preprocess(lvls, self.i_size, self.nc,
                                manufacture_obj_types.index("."))
        return self(inputs)


@gin.configurable
class ManufactureAugResnetRepairedMapAndOccupancy(
        WarehouseAugResnetRepairedMapAndOccupancy):
    """Resnet for predicting the agent cell occupancy (aka tile usage) on
    manufacture map.

    Args:
        i_size (int): size of input image.
        nc (int): number of input channels.
        ndf (int): number of output channels of conv2d layer.
        n_res_layers (int): number of residual layers (2x conv per residual
            layer).
        n_out (int): number of outputs.
    """

    def __init__(
        self,
        i_size: int = gin.REQUIRED,
        nc: int = gin.REQUIRED,
        ndf: int = gin.REQUIRED,
        n_res_layers: int = gin.REQUIRED,
        n_out: int = 1,
    ):
        super().__init__(
            i_size=i_size,
            nc=nc,
            ndf=ndf,
            n_res_layers=n_res_layers,
            n_out=n_out,
        )

    def int_logits_to_no_crop(self, lvls: torch.Tensor) -> torch.Tensor:
        inputs = int_preprocess(lvls, self.i_size, self.nc,
                                manufacture_obj_types.index("."))
        pred_repaired_map, occupancy = self(inputs)
        return pred_repaired_map, occupancy
