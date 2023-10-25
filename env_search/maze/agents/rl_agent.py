import os
from dataclasses import dataclass
from typing import Callable

import gin
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from ..envs.wrappers import VecMonitor, VecPreprocessImageWrapper
from .multigrid_network import MultigridNetwork


@gin.configurable
@dataclass
class RLAgentConfig:
    """Config for RLAgent."""

    recurrent_hidden_size: int = None
    """Size of the state used by the recurrent network."""

    model_path: str = None
    """Relative path to the saved model from a PAIRED/ACCEL run. Model assumed
    to be in `env_search/maze/agents/saved_models/{path}`."""

    n_envs: int = 10
    """Number of environments."""


@dataclass
class RLAgentResult:
    """Results from a simulation of an RL agent."""

    path_lengths: np.ndarray = None
    failed_list: np.ndarray = None
    aug_level: np.ndarray = None
    n_left_turns: np.ndarray = None
    n_right_turns: np.ndarray = None
    n_repeated_cells: int = None


class RLAgent:
    """Base class for RL agents solving mazes. Adapted from the open source
    codebase for PAIRED (https://github.com/ucl-dark/paired) and DCD
    (https://github.com/facebookresearch/dcd).

    Args:
        env_func: Function that creates a maze environment when called without
            any parameters. Either use gin to configure the env or use
            `functools.partial` to create a function with the config specified.
        n_evals: Number of evaluations.
        config: See `RLAgentConfig` for required configs.
    """

    def __init__(self, env_func: Callable, n_evals: int, config: RLAgentConfig):
        self.n_evals = n_evals
        self.recurrent_hidden_size = config.recurrent_hidden_size
        self.n_envs = config.n_envs

        env_fns = [env_func for _ in range(self.n_envs)]
        self.vec_env = DummyVecEnv(env_fns)

        # Wrap for modified infos
        self.vec_env = VecMonitor(venv=self.vec_env,
                                  filename=None,
                                  keep_buf=100)

        # Wrap for pre-processing the observations
        obs_key = "image"
        scale = 10.0
        transpose_order = [2, 0, 1]
        self.vec_env = VecPreprocessImageWrapper(
            venv=self.vec_env,
            obs_key=obs_key,
            transpose_order=transpose_order,
            scale=scale,
            device="cpu")

        num_directions = self.vec_env.observation_space["direction"].high[0] + 1
        self.model = MultigridNetwork(
            observation_space=self.vec_env.observation_space,
            action_space=self.vec_env.action_space,
            scalar_dim=num_directions,
            recurrent_hidden_size=config.recurrent_hidden_size,
        )

        model_path = os.path.join(os.path.dirname(__file__), "saved_models",
                                  config.model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)

    def eval_and_track(self,
                       level_shape: tuple,
                       aug_type: str = "agent_occupancy",
                       render_func: Callable = None) -> RLAgentResult:
        """
        Evaluate the agent and return the objectives and cell occupancy.
        Args:
            level_shape: Shape of the level.
            aug_type: One of
                "agent_occupancy" - Agent occupancy only,
                "turns" - Agent occupancy, left turn cells, right turn cells.
            render_func: A function that takes in the vec_env and does some
                rendering work.

        Returns:
            Array of objectives of length `self.n_evals` and the aug level
                matrix of the shape `level_shape` or `(3, *level_shape)`.
        """
        returns = []
        path_lengths = []
        n_left_turns = []
        n_right_turns = []
        failed_list = []

        obs = self.vec_env.reset()
        if render_func:
            render_func(self.vec_env)
        recurrent_hidden_states = (torch.zeros(self.n_envs,
                                               self.recurrent_hidden_size,
                                               device="cpu"),
                                   torch.zeros(self.n_envs,
                                               self.recurrent_hidden_size,
                                               device="cpu"))

        masks = torch.ones(1, device="cpu")

        if aug_type == "agent_occupancy":
            aug_level = np.zeros(level_shape, dtype=np.float)
        elif aug_type == "turns":
            aug_level = np.zeros((3, *level_shape), dtype=np.float)
        else:
            raise ValueError(f"Unknown aug_type: {aug_type}")

        left_turns = np.zeros(self.n_envs)
        right_turns = np.zeros(self.n_envs)
        aug_level_ind = np.zeros((self.n_envs, *level_shape))
        n_repeated_cells = 0
        while len(returns) < self.n_evals:
            xs = obs.get("x").detach().cpu().numpy().astype(int)
            ys = obs.get("y").detach().cpu().numpy().astype(int)
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = self.model.act(
                    obs, recurrent_hidden_states, masks)

            # Observe reward and next obs
            action = action.cpu().numpy()
            left_turns += (action == self.vec_env.envs[0].actions.left
                          ).flatten().astype(int)
            right_turns += (action == self.vec_env.envs[0].actions.right
                           ).flatten().astype(int)

            for i, (x, y) in enumerate(zip(xs, ys)):
                if aug_type == "agent_occupancy":
                    aug_level[y - 1,
                              x - 1] += 1  # Offset due to added outer walls
                    if aug_level_ind[i, y - 1, x - 1]:
                        n_repeated_cells += 1
                    else:
                        aug_level_ind[i, y - 1, x - 1] = 1
                elif aug_type == "turns":
                    aug_level[0, y - 1, x - 1] += 1
                    aug_level[1, y - 1, x - 1] += int(
                        action[i] == self.vec_env.envs[0].actions.left)
                    aug_level[2, y - 1, x - 1] += int(
                        action[i] == self.vec_env.envs[0].actions.right)

            obs, reward, done, infos = self.vec_env.step(action)

            if render_func:
                render_func(self.vec_env)

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device="cpu",
            )

            for i, info in enumerate(infos):
                if "episode" in info.keys():
                    returns.append(info["episode"]["r"])
                    path_lengths.append(info["episode"]["l"])
                    n_left_turns.append(left_turns[i])
                    n_right_turns.append(right_turns[i])
                    if returns[-1] > 0:
                        failed_list.append(0)
                    else:
                        failed_list.append(1)

                    # zero hidden states
                    recurrent_hidden_states[0][i].zero_()
                    recurrent_hidden_states[1][i].zero_()
                    left_turns[i] = 0
                    right_turns[i] = 0
                    aug_level_ind[i] = 0

                    if len(returns) >= self.n_evals:
                        break

        n_repeated_cells /= self.n_evals
        aug_level /= self.n_evals
        return RLAgentResult(
            np.array(path_lengths),
            np.array(failed_list),
            aug_level,
            np.array(n_left_turns),
            np.array(n_right_turns),
            n_repeated_cells,
        )
