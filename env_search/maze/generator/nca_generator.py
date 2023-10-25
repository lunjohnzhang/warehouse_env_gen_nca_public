import gin
import torch
import numpy as np

from torch import nn
from env_search.utils.network import int_preprocess_onehot
from env_search.utils import n_params


@gin.configurable
class MazeNCA(nn.Module):
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
        super().__init__()

        self.nc = nc
        self.n_aux_chan = n_aux_chan
        self._use_aux_chan = self.n_aux_chan > 0
        self.kernel_size = kernel_size
        self.n_hid_chan = n_hid_chan

        # We want the input and output to have the same W and H for each conv2d
        # layer, so we add padding.
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.padding = padding

        self.model = self._build_model(
            nc,
            n_aux_chan,
            kernel_size,
            padding,
            n_hid_chan,
        )
        self.aux_chans = None  # Initialize auxiliary memory as None

        # Remember number of params
        self.num_params = n_params(self)

    def _build_model(
        self,
        nc,
        n_aux_chan,
        kernel_size,
        padding,
        n_hid_chan,
    ):
        model = nn.Sequential()

        # Three layers of conv2d
        self.n_in_chan = nc + n_aux_chan
        model.add_module(
            f"initial:conv:in_chan-{n_hid_chan}",
            nn.Conv2d(
                self.n_in_chan,
                n_hid_chan,
                kernel_size,
                1,
                padding,
                bias=True,
            ),
        )
        model.add_module(f"initial:relu", nn.ReLU(inplace=True))

        model.add_module(
            f"internal1:conv:{n_hid_chan}-{n_hid_chan}",
            nn.Conv2d(n_hid_chan, n_hid_chan, 1, 1, 0, bias=True),
        )
        model.add_module(f"internal1:relu", nn.ReLU(inplace=True))

        model.add_module(
            f"internal2:conv:{n_hid_chan}-{self.n_in_chan}",
            nn.Conv2d(n_hid_chan, self.n_in_chan, 1, 1, 0, bias=True),
        )
        model.add_module("internal2:sigmoid", nn.Sigmoid())

        return model

    def _extract_layout(self, inputs):
        """Helper function to extract layout from raw NCA output"""

        if self._use_aux_chan:
            inputs = inputs[:, :-self.n_aux_chan, :, :]
            self.aux_chans = inputs[:, -self.n_aux_chan:, :, :]

        inputs = torch.argmax(inputs, dim=1)
        return inputs

    def generate(self, seed_envs, n_iter, save=False):
        """Generate maze env with the model using seed_env for n_iters

        Args:
            seed_envs (n, lvl_height, lvl_width): tensor of int levels
            n_iter (int): number of iterations
            save (bool): whether to save itermediate steps
        """

        batch_size, lvl_height, lvl_width = seed_envs.shape

        inputs = int_preprocess_onehot(seed_envs, self.nc)

        all_sols = None
        if save:
            all_sols = []

        # Add auxiliary memory
        if self._use_aux_chan:
            if self.aux_chans is None:
                self.aux_chans = torch.zeros(
                    size=(
                        batch_size,
                        self.n_aux_chan,
                        lvl_height,
                        lvl_width,
                    ),
                    device=inputs.device,
                )
            inputs = torch.cat([inputs, self.aux_chans], axis=1)

        with torch.no_grad():
            for _ in range(n_iter):
                inputs = self.forward(inputs)
                if save:
                    inter_layout = self._extract_layout(inputs)
                    all_sols.append(inter_layout.squeeze().cpu().numpy())

        # if self._use_aux_chan:
        #     inputs = inputs[:, :-self.n_aux_chan, :, :]
        #     self.aux_chans = inputs[:, -self.n_aux_chan:, :, :]

        # inputs = torch.argmax(inputs, dim=1)
        inputs = self._extract_layout(inputs)

        return inputs, all_sols

    def forward(self, inputs):
        """Runs the network on input once"""
        return self.model(inputs)

    def set_params(self, weights):
        """Set the params of the model

        Args:
            weights (np.ndarray): weights to set, 1D numpy array
        """
        with torch.no_grad():
            assert weights.shape == (self.num_params, )

            state_dict = self.model.state_dict()

            s_idx = 0
            for param_name in state_dict:
                param_shape = state_dict[param_name].shape
                param_dtype = state_dict[param_name].dtype
                param_device = state_dict[param_name].device
                curr_n_param = np.prod(param_shape)
                to_set = torch.tensor(
                    weights[s_idx:s_idx + curr_n_param],
                    dtype=param_dtype,
                    requires_grad=True,  # May used by dqd
                    device=param_device,
                )
                to_set = torch.reshape(to_set, param_shape)
                assert to_set.shape == param_shape
                s_idx += curr_n_param
                state_dict[param_name] = to_set

            # Load new params
            self.model.load_state_dict(state_dict)

            # Reset aux_chan
            self.aux_chans = None