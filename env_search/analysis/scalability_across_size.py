import os
import yaml
import fire
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from env_search.analysis.utils import get_color

mpl.use("agg")
plt.rc("text", usetex=True)


def sort_logdir_size(all_sizes):
    all_sizes_pair = {
        size: np.prod([int(a) for a in size.split("x")])
        for size in all_sizes
    }
    all_sizes_pair = sorted(all_sizes_pair.items(), key=lambda item: item[1])
    all_sizes = [size[0] for size in all_sizes_pair]
    return all_sizes


def scalability_across_size(
    logdirs_across_sizes,
    add_legend=False,
    front_fig=False,
):
    """
    Plot env size vs. scalability plot. This function assumes the numerical
    result is already gotten by running `throughput_vs_n_agents`.
    """
    all_sizes = []
    for logdir_size in os.listdir(logdirs_across_sizes):
        logdir_size_full = os.path.join(logdirs_across_sizes, logdir_size)
        if not os.path.isdir(logdir_size_full):
            continue
        all_sizes.append(logdir_size)

    all_sizes = sort_logdir_size(all_sizes)
    all_max_scalability = {}
    all_max_throughput = {}
    all_color = {}
    for logdir_size in all_sizes:
        logdir_size_full = os.path.join(logdirs_across_sizes, logdir_size)
        for logdirs_plot in os.listdir(logdir_size_full):
            logdirs_plot_full = os.path.join(logdir_size_full, logdirs_plot)
            if not os.path.isdir(logdirs_plot_full):
                continue

            # Read in meta file
            with open(os.path.join(logdirs_plot_full, "meta.yaml"), "r") as f:
                meta = yaml.safe_load(f)
            algo_name = meta["algorithm"]
            map_size = meta["map_size"]
            mode = meta["mode"]
            map_from = meta["map_from"]

            # Read in numerical result and get max scalability
            numerical_file = os.path.join(
                logdirs_plot_full,
                f"numerical_{algo_name}_{map_size}_{mode}.csv")
            numerical_result_df = pd.read_csv(numerical_file)
            max_throughput_idx = numerical_result_df["mean_throughput"].idxmax(
            )
            max_scalability = numerical_result_df.iloc[max_throughput_idx][
                "agent_num"]
            max_throughput = numerical_result_df.iloc[max_throughput_idx][
                "mean_throughput"]

            color = get_color(map_from, algo_name)

            # For front fig plotting, change color
            # if front_fig and map_from == "CMA-MAE + NCA (alpha=5)":
            #     color = "orange"

            # Add to result dict
            if map_from in all_max_scalability:
                all_max_scalability[map_from].append(max_scalability)
            else:
                all_max_scalability[map_from] = [max_scalability]

            if map_from in all_max_throughput:
                all_max_throughput[map_from].append(max_throughput)
            else:
                all_max_throughput[map_from] = [max_throughput]

            all_color[map_from] = color

    figsize = (18, 8) if front_fig and add_legend else (8, 8)
    fig, ax_scalability = plt.subplots(1, 1, figsize=figsize)

    # Adding Twin Axes to plot using throughput
    ax_throughput = ax_scalability.twinx()

    for map_from in all_max_scalability:
        max_scalability = all_max_scalability[map_from]
        max_throughput = all_max_throughput[map_from]
        ax_scalability.plot(max_scalability,
                            label=map_from.replace("alpha", r"$\alpha$"),
                            marker=".",
                            markersize=20,
                            color=all_color[map_from])
        ax_throughput.plot(max_throughput,
                           label=map_from.replace("alpha", r"$\alpha$"),
                           marker="^",
                           markersize=15,
                           color=all_color[map_from])

    # Create custom handles and labels that separate lines (algo) and marker
    # (scalability or throughout)
    handles = []
    labels = []
    for map_from, color in all_color.items():
        handles.append(
            Line2D([0], [0],
                   color=color,
                   lw=4,
                   label=map_from.replace("alpha", r"$\alpha$")))
        if front_fig:
            if map_from == "Human":
                labels.append("Human-designed Map")
            elif map_from == "CMA-MAE + NCA (alpha=5)":
                labels.append("Optimized Map")
        else:
            labels.append(map_from.replace("alpha", r"$\alpha$"))
    # For markers
    handles.append(
        Line2D([0], [0],
               marker='.',
               color='w',
               markerfacecolor='k',
               label='Max Scalability',
               markersize=30))
    handles.append(
        Line2D([0], [0],
               marker='^',
               color='w',
               markerfacecolor='k',
               label='Max Throughput',
               markersize=25))
    labels.extend(['Max Scalability', 'Max Throughput'])

    # handles, labels = ax_scalability.get_legend_handles_labels()

    if add_legend:
        legend = ax_scalability.legend(
            handles,
            labels,
            loc="lower left",
            ncol=2,
            fontsize=30,
            mode="expand",
            bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
            # borderaxespad=0,)
        )
    ax_scalability.tick_params(axis='y', labelsize=32)
    ax_scalability.set_xticks(np.arange(len(all_sizes)))
    ax_scalability.set_xticklabels(all_sizes, fontsize=30)
    ax_scalability.set_xlabel("Environment Size", fontsize=45)
    ax_scalability.set_ylabel("Max Scalability", fontsize=45)

    ax_throughput.tick_params(axis='y', labelsize=32)
    ax_throughput.set_ylabel("Max Throughput", fontsize=45)

    fig.savefig(
        os.path.join(
            logdirs_across_sizes,
            f"scalability.pdf",
        ),
        dpi=300,
        bbox_inches='tight',
    )

    fig.savefig(
        os.path.join(
            logdirs_across_sizes,
            f"scalability.png",
        ),
        dpi=300,
        bbox_inches='tight',
    )


if __name__ == "__main__":
    fire.Fire(scalability_across_size)
