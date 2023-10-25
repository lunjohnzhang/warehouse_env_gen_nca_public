"""Class representing the results of an evaluation."""
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


def maybe_mean(arr, indices=None):
    """Calculates mean of arr[indices] if possible.

    indices should be a list. If it is None, the mean of the whole arr is taken.
    """
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.mean(arr[indices], axis=0)


def maybe_median(arr, indices=None):
    """Same as maybe_mean but with median."""
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.median(arr[indices], axis=0)


def maybe_std(arr, indices=None):
    """Same as maybe_mean but with std."""
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.std(arr[indices], axis=0)


@dataclass
class MazeMetadata:
    """Metadata obtained by running maze envs"""

    level: np.ndarray = None  # Generated level
    objs: List = None  # Objectives: Currently [0] or [1] based on solvability

    # Env-based measures
    solvability: int = None  # 1 if solvable, 0 otherwise
    num_blocks: int = None  # Number of filled cells in the maze
    optimal_path_length: int = None  # -1 if unsolvable
    layout_entropy: float = None

    # Agent-based measures (the module aggregates it for now)
    agent_path_length: int = None
    n_left_turns: int = None
    n_right_turns: int = None
    n_repeated_cells: int = None
    frac_explored_cells: int = None

    # Misc
    aug_level: np.ndarray = None  # Level with the longest optimal path


@dataclass
class MazeResult:  # pylint: disable = too-many-instance-attributes
    """Represents `n` results from an objective function evaluation.

    `n` is typically the number of evals (n_evals).

    Different fields are filled based on the objective function.
    """

    ## Raw data ##

    maze_metadata: dict = None

    ## Aggregate data ##

    agg_obj: float = None
    agg_result_obj: float = None
    agg_measures: np.ndarray = None  # (behavior_dim,) array

    ## Measures of spread ##

    std_obj: float = None
    std_measure: np.ndarray = None  # (behavior_dim,) array

    ## Other data ##

    failed: bool = False
    log_message: str = None

    @staticmethod
    def from_raw(
        maze_metadata: MazeMetadata,
        opts: dict = None,
    ):
        """Constructs a MazeResult from raw data.

        `opts` is a dict with several configuration options. It may be better as
        a gin parameter, but since MazeResult is created on workers, gin
        parameters are unavailable (unless we start loading gin on workers too).
        Options in `opts` are:

            `measure_names`: Names of the measures to return
            `aggregation` (default="mean"): How each piece of data should be
                aggregated into single values. Options are:
                - "mean": Take the mean, e.g. mean measure
                - "median": Take the median, e.g. median measure (element-wise)
        """
        # Handle config options.
        opts = opts or {}
        if "measure_names" not in opts:
            raise ValueError("opts should contain `measure_names`")

        opts.setdefault("aggregation", "mean")

        # For maze, obj and obj_result are the same
        if opts["aggregation"] == "mean":
            agg_obj = maybe_mean(maze_metadata.objs)
            agg_result_obj = maybe_mean(maze_metadata.objs)
        elif opts["aggregation"] == "median":
            agg_obj = maybe_median(maze_metadata.objs)
            agg_result_obj = maybe_median(maze_metadata.objs)
        else:
            raise ValueError(f"Unknown aggregation {opts['aggregation']}")

        agg_measures = MazeResult._obtain_measure_values(
            asdict(maze_metadata), opts["measure_names"])

        return MazeResult(
            maze_metadata=asdict(maze_metadata),
            agg_obj=agg_obj,
            agg_result_obj=agg_result_obj,
            agg_measures=agg_measures,
            # std_obj=maybe_std(objs, std_indices),
            # std_measure=maybe_std(measures, std_indices),
        )

    @staticmethod
    def _obtain_measure_values(metadata, measure_names):
        measures = []
        for measure_name in measure_names:
            measure_val = metadata[measure_name]
            measures.append(measure_val)

        return np.array(measures)
