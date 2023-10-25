"""Miscellaneous project-wide utilities."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from env_search import MAP_DIR


# 6 object types for kiva map:
# '.' (0) : empty space
# '@' (1): obstacle (shelf)
# 'e' (2): endpoint (point around shelf)
# 'r' (3): robot start location (not searched)
# 's' (4): one of 'r'
# 'w' (5): workstation
# NOTE:
# 1: only the first 2 or 3 objects are searched by QD
# 2: s (r_s) is essentially one of r s.t. in milp can make the graph
# connected
kiva_obj_types = ".@ersw"
KIVA_ROBOT_BLOCK_WIDTH = 4
KIVA_WORKSTATION_BLOCK_WIDTH = 2
KIVA_ROBOT_BLOCK_HEIGHT = 4
MIN_SCORE = 0

# 6 object types for kiva map:
# '.' (0) : empty space
# '0' (1): manufacture station type 0 (obstacle)
# '1' (2): manufacture station type 1 (obstacle)
# '2' (3): manufacture station type 2 (obstacle)
# 'e' (4): endpoint (point around manufacture station)
# 's' (5): special 'e'.
# NOTE:
# 1: only the first 5 objects are searched by QD
# 2: s (e_s) is essentially one of e s.t. in milp can make the graph connected
manufacture_obj_types = ".012es"

# 2 object types for maze map
# ' ' (0): empty space
# 'X' (1): obstacle
maze_obj_types = " X"


def format_env_str(env_str):
    """Format the env from List[str] to pure string separated by \n """
    return "\n".join(env_str)

def env_str2number(env_str, obj_types):
    env_np = []
    for row_str in env_str:
        # print(row_str)
        row_np = [obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)


def env_number2str(env_np, obj_types):
    env_str = []
    n_row, n_col = env_np.shape
    for i in range(n_row):
        curr_row = []
        for j in range(n_col):
            curr_row.append(obj_types[env_np[i, j]])
        env_str.append("".join(curr_row))
    return env_str


def kiva_env_str2number(env_str):
    """
    Convert kiva env in string format to np int array format.

    Args:
        env_str (List[str]): kiva env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, kiva_obj_types)


def kiva_env_number2str(env_np):
    """
    Convert kiva env in np int array format to str format.

    Args:
        env_np (np.ndarray): kiva env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, kiva_obj_types)


def manufacture_env_str2number(env_str):
    """
    Convert manufacture env in string format to np int array format.

    Args:
        env_str (List[str]): manufacture env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, manufacture_obj_types)


def manufacture_env_number2str(env_np):
    """
    Convert manufacture env in np int array format to str format.

    Args:
        env_np (np.ndarray): manufacture env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, manufacture_obj_types)

def maze_env_str2number(env_str):
    """
    Convert maze env in string format to np int array format.

    Args:
        env_str (List[str]): maze env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, maze_obj_types)


def maze_env_number2str(env_np):
    """
    Convert maze env in np int array format to str format.

    Args:
        env_np (np.ndarray): maze env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, maze_obj_types)

def flip_one_r_to_s(env_np, obj_types=kiva_obj_types):
    """
    Change one of 'r' in the env to 's' for milp
    """
    all_r = np.argwhere(env_np == obj_types.index("r"))
    if len(all_r) == 0:
        raise ValueError("No 'r' found")
    to_replace = all_r[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np

def flip_one_e_to_s(env_np, obj_types=kiva_obj_types):
    """
    Change one of 'e' in the env to 's' for milp
    """
    all_e = np.argwhere(env_np == obj_types.index("e"))
    if len(all_e) == 0:
        raise ValueError("No 'e' found")
    to_replace = all_e[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np

def flip_tiles(env_np, from_tile, to_tile, obj_types = kiva_obj_types):
    """Replace ALL occurance of `from_tile` to `flip_target` in `to_tile"""
    all_from_tiles = np.where(env_np == obj_types.index(from_tile))
    if len(all_from_tiles[0]) == 0:
        raise ValueError(f"No '{from_tile}' found")
    env_np[all_from_tiles] = obj_types.index(to_tile)
    return env_np

def flip_tiles_torch(env_torch, from_tile, to_tile, obj_types = kiva_obj_types):
    """Replace ALL occurance of `from_tile` to `flip_target` in `to_tile"""
    all_from_tiles = torch.where(env_torch == obj_types.index(from_tile))
    if len(all_from_tiles[0]) == 0:
        raise ValueError(f"No '{from_tile}' found")
    env_torch[all_from_tiles] = obj_types.index(to_tile)
    return env_torch

def read_in_kiva_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name

def read_in_manufacture_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name

def read_in_maze_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name

def write_map_str_to_json(
    map_filepath,
    map_str,
    name,
    domain,
    nca_runtime=None,
    milp_runtime=None,
    nca_milp_runtime=None,
    sim_score=None,
):
    to_write = {
        "name": name,
        "layout": map_str,
        "nca_runtime": nca_runtime,
        "milp_runtime": milp_runtime,
        "nca_milp_runtime": nca_milp_runtime,
        "sim_score": sim_score,
    }
    if domain == "manufacture":
        map_np = manufacture_env_str2number(map_str)
        to_write["weight"] = False
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]
        to_write["n_0"] = sum(row.count('0') for row in map_str)
        to_write["n_1"] = sum(row.count('1') for row in map_str)
        to_write["n_2"] = sum(row.count('2') for row in map_str)
        to_write["n_stations"] = \
            to_write["n_0"] + to_write["n_1"] + to_write["n_2"]
    elif domain == "kiva":
        map_np = kiva_env_str2number(map_str)
        to_write["weight"] = False
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]
        to_write["n_endpoint"] = sum(row.count('e') for row in map_str)
        to_write["n_agent_loc"] = sum(row.count('r') for row in map_str)
        to_write["n_shelf"] = sum(row.count('@') for row in map_str)
        to_write["maxtime"] = 5000
    elif domain == "maze":
        map_np = maze_env_str2number(map_str)
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]

    with open(map_filepath, "w") as json_file:
        json.dump(to_write, json_file, indent=4)

def set_spines_visible(ax: plt.Axes):
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(True)


def n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # params_ = 0
    # state_dict = model.state_dict()
    # for _, param in state_dict.items():
    #     params_ += np.prod(param.shape)
    # print("validate: ", params_)

    return params

def rewrite_map(path, domain):
    if domain == "manufacture":
        env, name = read_in_manufacture_map(path)
    elif domain == "kiva":
        env, name = read_in_kiva_map(path)
    write_map_str_to_json(path, env, name, domain)