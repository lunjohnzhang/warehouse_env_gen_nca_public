import os
import fire
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
import imageio
import moviepy.editor as mp

from PIL import Image
from scipy.sparse import csgraph
from functools import partial
from matplotlib import colors

from env_search import MAP_DIR
from env_search.utils import (
    kiva_obj_types,
    kiva_env_str2number,
    kiva_env_number2str,
    read_in_kiva_map,
    read_in_manufacture_map,
    KIVA_ROBOT_BLOCK_WIDTH,
    manufacture_env_str2number,
    read_in_maze_map,
)
from env_search.utils import set_spines_visible
from env_search.maze.level import OBJ_TYPES_TO_INT, MazeLevel
from env_search.maze.module import MazeModule
from env_search.maze.agents.rl_agent import RLAgent, RLAgentConfig
from env_search.maze.envs.maze import MazeEnv


FIG_HEIGHT = 10

def convert_avi_to_gif(
        input_path,
        output_path,
        output_resolution=(640, 640),
):
    # Load the input video file
    clip = mp.VideoFileClip(input_path)

    # Convert the video to a sequence of frames
    frames = []
    for frame in clip.iter_frames():
        # Resize the frame to the desired output resolution
        frame = Image.fromarray(frame).resize(output_resolution)
        frames.append(frame)

    # Write the frames to a GIF file
    # imageio.mimsave(output_path, frames, fps=clip.fps, size=output_resolution)
    imageio.mimsave(
        output_path,
        frames,
        fps=clip.fps,
        format='gif',
        palettesize=256,
    )


def create_movie(folder_path, filename):
    glob_str = os.path.join(folder_path, '*.png')
    image_files = sorted(glob.glob(glob_str))

    # Grab the dimensions of the image
    img = cv2.imread(image_files[0])
    image_dims = img.shape[:2][::-1]

    # Create a video
    avi_output_path = os.path.join(folder_path, f"{filename}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 30
    video = cv2.VideoWriter(
        avi_output_path,
        fourcc,
        frame_rate,
        image_dims,
    )

    for img_filename in image_files:
        img = cv2.imread(img_filename)
        video.write(img)

    video.release()

    # Convert video to gif
    gif_output_path = os.path.join(folder_path, f"{filename}.gif")
    convert_avi_to_gif(avi_output_path, gif_output_path, image_dims)


def visualize_env(env_np, cmap, norm, ax, fig, save, filenames, store_dir, dpi):
    # heatmap = plt.pcolor(np.array(data), cmap=cmap, norm=norm)
    # plt.colorbar(heatmap, ticks=[0, 1, 2, 3])
    sns.heatmap(
        env_np,
        square=True,
        cmap=cmap,
        norm=norm,
        ax=ax,
        cbar=False,
        rasterized=True,
        annot_kws={"size": 30},
        linewidths=1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
    )

    set_spines_visible(ax)
    ax.figure.tight_layout()

    if save:
        ax.margins(x=0, y=0)
        for filename in filenames:
            fig.savefig(
                os.path.join(store_dir, filename),
                dpi=dpi,
                bbox_inches='tight',
                # pad_inches=0,
                rasterized=True,
            )
        plt.close('all')


def visualize_kiva(
    env_np: np.ndarray,
    filenames: List = None,
    store_dir: str = MAP_DIR,
    dpi: int = 300,
    ax: plt.Axes = None,
    figsize: tuple = None,
):
    """
    Visualize kiva layout. Will store image under `store_dir`

    Args:
        env_np: layout in numpy format
    """
    n_row, n_col = env_np.shape
    save = False
    if ax is None:
        if figsize is None:
            # figsize = (n_col, n_row)
            figsize = (FIG_HEIGHT * n_col/n_row, FIG_HEIGHT)
            # if n_col > 50 or n_row > 50:
            #     figsize = (16, 16)
        # fig, ax = plt.subplots(1, 1, figsize=(n_col, n_row))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        save = True
    else:
        fig = ax.get_figure()
    cmap = colors.ListedColormap(
        ['white', 'black', 'deepskyblue', 'orange', 'fuchsia'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    visualize_env(env_np, cmap, norm, ax, fig, save, filenames, store_dir, dpi)


def visualize_manufacture(
    env_np: np.ndarray,
    filenames: List = None,
    store_dir: str = MAP_DIR,
    dpi: int = 300,
    ax: plt.Axes = None,
    figsize: tuple = None,
):
    """
    Visualize manufacture layout. Will store image under `store_dir`

    Args:
        env_np: layout in numpy format
    """
    n_row, n_col = env_np.shape
    save = False
    if ax is None:
        if figsize is None:
            # figsize = (n_col, n_row)
            # if n_col > 50 or n_row > 50:
            figsize = (FIG_HEIGHT * n_col/n_row, FIG_HEIGHT)
        # fig, ax = plt.subplots(1, 1, figsize=(n_col, n_row))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        save = True
    else:
        fig = ax.get_figure()
    cmap = colors.ListedColormap([
        'white',
        'red', # construction station 0
        'forestgreen', # construction station 1
        'gold', # construction station 2
        'deepskyblue', # endpoint
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    visualize_env(env_np, cmap, norm, ax, fig, save, filenames, store_dir, dpi)


def post_process_maze_env(env_np: np.ndarray):
    # print("==> Level <==")
    # print(MazeLevel(env_np).to_str())
    # print("Bit map:")
    # print(env_np)
    adj = MazeModule._get_adj(env_np)

    # Find the best distances
    dist, predecessors = csgraph.floyd_warshall(adj, return_predecessors=True)
    dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

    # print(f"Optimal path length: {dist.max()}")
    # Label the start and the end point
    endpoints = np.unravel_index(dist.argmax(), dist.shape)
    start_cell, end_cell = zip(*np.unravel_index(endpoints, env_np.shape))

    endpoint_level = env_np.copy()
    endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
    endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

    return endpoint_level, start_cell, end_cell


def visualize_maze(
    env_np: np.ndarray,
    filenames: List = None,
    store_dir: str = MAP_DIR,
):
    endpoint_level, start_cell, end_cell = post_process_maze_env(env_np)

    # Offset start, goal to account for the added outer walls
    start_pos = (start_cell[1] + 1, start_cell[0] + 1)
    goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
    # print(f"Start: {start_pos}; End: {goal_pos}")
    env_func = partial(MazeEnv,
                       size=env_np.shape[0] + 2,
                       bit_map=env_np,
                       start_pos=start_pos,
                       goal_pos=goal_pos)
    rl_agent_conf = RLAgentConfig(recurrent_hidden_size=256,
                                  model_path="accel_seed_1/model_20000.tar",
                                  n_envs=1)
    rl_agent = RLAgent(env_func, n_evals=1, config=rl_agent_conf)
    rl_agent.vec_env.reset()
    img = rl_agent.vec_env.render(mode="rgb_array")
    for filename in filenames:
        name = os.path.join(store_dir, filename)
        cv2.imwrite(name, img)
        print(f"Saved to {name}")


def main(map_filepath, store_dir=MAP_DIR, domain="kiva"):
    """
    Args:
        domain: one of ['kiva', 'manufacture', 'maze']
    """
    if domain == "kiva":
        kiva_map, map_name = read_in_kiva_map(map_filepath)
        visualize_kiva(kiva_env_str2number(kiva_map),
                       store_dir=store_dir,
                       filenames=[f"{map_name}.png"])
    elif domain == "manufacture":
        manufacture_map, map_name = read_in_manufacture_map(map_filepath)
        visualize_manufacture(manufacture_env_str2number(manufacture_map),
                              store_dir=store_dir,
                              filenames=[f"{map_name}.png"])
    elif domain == "maze":
        maze_map, map_name = read_in_maze_map(map_filepath)
        visualize_maze(MazeLevel.str_to_number(maze_map).astype(int),
                       store_dir=store_dir,
                       filenames=[f"{map_name}.png"])


if __name__ == '__main__':
    fire.Fire(main)