"""Code from https://github.com/ucl-dark/paired.
Provides wrappers for vectorized envs."""
import time
from collections import deque

import gym
import numpy as np
import torch
from stable_baselines3.common.monitor import ResultsWriter
from stable_baselines3.common.vec_env import VecEnvWrapper


class VecPreprocessImageWrapper(VecEnvWrapper):

    def __init__(
        self,
        venv,
        obs_key=None,
        transpose_order=None,
        scale=None,
        channel_first=False,
        to_tensor=True,
        device=None,
    ):
        super().__init__(venv)

        self.is_dict_obs = isinstance(venv.observation_space, gym.spaces.Dict)

        self.transpose_order = transpose_order
        if self.transpose_order:
            self.batch_transpose_order = [
                0,
            ] + list([i + 1 for i in transpose_order])
        else:
            self.batch_transpose_order = None
        self.obs_key = obs_key
        self._obs_space = None
        self._adversary_obs_space = None
        self.to_tensor = to_tensor
        self.device = device

        # Colorspace parameters
        self.scale = scale
        self.channel_first = channel_first
        self.channel_index = 1 if channel_first else -1

        image_obs_space = self.venv.observation_space
        if self.obs_key is not None and "," in self.obs_key:
            self.obs_key = self.obs_key.split(",")
        if self.obs_key and not isinstance(self.obs_key, list):
            image_obs_space = image_obs_space[self.obs_key]
            self.num_channels = image_obs_space.shape[self.channel_index]

        delattr(self, "observation_space")

    def _obs_dict_to_tensor(self, obs):
        for k in obs.keys():
            if isinstance(obs[k], np.ndarray):
                obs[k] = torch.from_numpy(obs[k]).float()
                if self.device:
                    obs[k] = obs[k].to(self.device)
        return obs

    def _transpose(self, obs):
        if len(obs.shape) == len(self.batch_transpose_order):
            return obs.transpose(*self.batch_transpose_order)
        else:
            return obs.transpose(*self.transpose_order)

    def _preprocess(self, obs, obs_key=None):
        if obs_key is None:
            if self.scale:
                obs = obs / self.scale

            if self.batch_transpose_order:
                obs = self._transpose(obs)

            if isinstance(obs, np.ndarray) and self.to_tensor:
                obs = torch.from_numpy(obs).float()
                if self.device:
                    obs = obs.to(self.device)
            elif isinstance(obs, dict) and self.to_tensor:
                obs = self._obs_dict_to_tensor(obs)
        elif isinstance(obs_key, list):
            for key in obs_key:
                if self.scale:
                    obs[key] = obs[key] / self.scale

                if self.batch_transpose_order:
                    obs[key] = self._transpose(obs[key])
                    if "full_obs" in obs:
                        obs["full_obs"] = self._transpose(obs["full_obs"])

                if self.to_tensor:
                    obs = self._obs_dict_to_tensor(obs)
        else:
            if self.scale:
                obs[self.obs_key] = obs[self.obs_key] / self.scale

            if self.batch_transpose_order:
                obs[self.obs_key] = self._transpose(obs[self.obs_key])
                if "full_obs" in obs:
                    obs["full_obs"] = self._transpose(obs["full_obs"])

            if self.to_tensor:
                obs = self._obs_dict_to_tensor(obs)

        return obs

    def _transpose_box_space(self, space):
        if isinstance(space, gym.spaces.Box):
            shape = np.array(space.shape)
            shape = shape[self.transpose_order]
            return gym.spaces.Box(low=0, high=255, shape=shape, dtype="uint8")
        else:
            raise ValueError("Expected gym.spaces.Box")

    def _transpose_obs_space(self, obs_space):
        if self.obs_key:
            if isinstance(obs_space, gym.spaces.Dict):
                keys = obs_space.spaces
            else:
                keys = obs_space.keys()
            transposed_obs_space = {k: obs_space[k] for k in keys}
            transposed_obs_space[self.obs_key] = self._transpose_box_space(
                transposed_obs_space[self.obs_key])

            if "full_obs" in transposed_obs_space:
                transposed_obs_space["full_obs"] = self._transpose_box_space(
                    transposed_obs_space["full_obs"])
        else:
            transposed_obs_space = self._transpose_box_space(obs_space)

        return transposed_obs_space

    # Public interface
    def reset(self):
        obs = self.venv.reset()
        return self._preprocess(obs, obs_key=self.obs_key)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = self._preprocess(obs, obs_key=self.obs_key)

        for i, info in enumerate(infos):
            if "truncated_obs" in info:
                truncated_obs = info["truncated_obs"]
                infos[i]["truncated_obs"] = self._preprocess(
                    truncated_obs, obs_key=self.obs_key)

        if self.to_tensor:
            rews = torch.from_numpy(rews).unsqueeze(dim=1).float()

        return obs, rews, dones, infos

    def get_observation_space(self):
        if self._obs_space:
            return self._obs_space

        obs_space = self.venv.observation_space

        if self.batch_transpose_order:
            self._obs_space = self._transpose_obs_space(obs_space)
        else:
            self._obs_space = obs_space

        return self._obs_space

    def __getattr__(self, name):
        if name == "observation_space":
            return self.get_observation_space()
        else:
            return getattr(self.venv, name)


class VecMonitor(VecEnvWrapper):

    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename,
                                                header={"t_start": self.tstart},
                                                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, "f")
        self.eplens = np.zeros(self.num_envs, "i")
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {
                    "r": ret,
                    "l": eplen,
                    "t": round(time.time() - self.tstart, 6)
                }
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info["episode"] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info

        return obs, rews, dones, newinfos
