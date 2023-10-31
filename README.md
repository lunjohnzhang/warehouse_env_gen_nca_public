# Arbitrarily Scalable Environment Generators via Neural Cellular Automata

This repository is the official implementation of **[Arbitrarily Scalable Environment Generators via Neural Cellular Automata](https://arxiv.org/abs/2310.18622)** published in NeurIPS 2023. The repository builds on top of the repository of [Multi-Robot Coordination and Layout Design for Automated Warehousing](https://github.com/lunjohnzhang/warehouse_env_gen_public).

## Installation

This is a hybrid C++/Python project. The simulation environment is written in C++ and the rests are in Python. We use [pybind11](https://pybind11.readthedocs.io/en/stable/) to bind the two languages.

1. **Initialize pybind11:** After cloning the repo, initialize the pybind11 submodule

   ```bash
   git submodule init
   git submodule update
   ```

1. **Install Singularity:** All of our code runs in a Singularity container.
   Singularity is a container platform (similar in many ways to Docker). Please
   see the instructions
   [here](https://sylabs.io/singularity/) for installing SingularityCE. As a reference, we use version 3.11.1.

1. **Download Boost:** From the root directory of the project, run the following to download the Boost 1.71, which is required for compiling C++ simulator. You don't have to install it on your system since it will be passed into the container and installed there.

   ```
   wget https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz --no-check-certificate
   ```

1. **Install CPLEX:** CPLEX is used for repairing the generated warehouse maps.

   1. Download the free academic version [here](https://www.ibm.com/products/ilog-cplex-optimization-studio).
   2. Download the installation file for Linux.
   3. Follow this [guide](https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio) to install it. Basically:

   ```
   chmod u+x INSTALLATION_FILE
   ./INSTALLATION_FILE
   ```

   During installation, set the installation directory to `CPLEX_Studio2210/` in the repo.

1. **Build Singularity container:** Run the provided script to build the container. Note that this need `sudo` permission on your system.
   ```
   bash build_container.sh
   ```
   The script will first build a container as a sandbox, compile the C++ simulator, then convert that to a regular `.sif` Singularity container.

### For user in mainland China

If you are in mainland China, please use the corresponding `build_container_cn.sh` script to build the singularity container. It will use the local mirrors in China to download relevant packages (mainly debian and python), which is much faster.

## Optimizing Environments

### Training Logging Directory Manifest

Regardless of where the script is run, the log files and results are placed in a
logging directory in `logs/`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_<dashed-name>_<uuid>`, e.g.
`2020-12-01_15-00-30_experiment-1_ff1dcb2b`. Inside each directory are the
following files:

```text
- config.gin  # All experiment config variables, lumped into one file.
- seed  # Text file containing the seed for the experiment.
- reload.pkl  # Data necessary to reload the experiment if it fails.
- reload_em.pkl  # Pickle data for EmulationModel.
- reload_em.pth  # PyTorch models for EmulationModel.
- metrics.json  # Data for a MetricLogger with info from the entire run, e.g. QD score.
- hpc_config.sh  # Same as the config in the Slurm dir, if Slurm is used.
- archive/  # Snapshots of the full archive, including solutions and metadata,
            # in pickle format.
- archive_history.pkl  # Stores objective values and behavior values necessary
                       # to reconstruct the archive. Solutions and metadata are
                       # excluded to save memory.
- dashboard_status.txt  # Job status which can be picked up by dashboard scripts.
                        # Only used during execution.
- evaluations # Output logs of LMAPF simulator
```

### Running Locally

#### Single Run

To run one experiment locally, use:

```bash
bash scripts/run_local.sh CONFIG SEED NUM_WORKERS
```

For instance, with 4 workers:

```bash
bash scripts/run_local.sh config/foo.gin 42 4
```

`CONFIG` is the [gin](https://github.com/google/gin-config) experiment config
for `env_search/main.py`.

### Running on Slurm

Use the following command to run an experiment on an HPC with Slurm (and
Singularity) installed:

```bash
bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG
```

For example:

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh
```

`CONFIG` is the experiment config for `env_search/main.py`, and `HPC_CONFIG` is a shell
file that is sourced by the script to provide configuration for the Slurm
cluster. See `config/hpc` for example files.

Once the script has run, it will output commands like the following:

- `tail -f ...` - You can use this to monitor stdout and stderr of the main
  experiment script. Run it.
- `bash scripts/slurm_cancel.sh ...` - This will cancel the job.

### Reloading

While the experiment is running, its state is saved to `reload.pkl` in the
logging directory. If the experiment fails, e.g. due to memory limits, time
limits, or network connection issues, `reload.pkl` may be used to continue the
experiment. To do so, execute the same command as before, but append the path to
the logging directory of the failed experiment.

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh -r logs/.../
```

The experiment will then run to completion in the same logging directory. This
works with `scripts/run_local.sh` too.

### Environment Optimization Using QD Algorithms

The `config/` directory contains the config files required to run the experiments to either train the NCA generators uing CMA-MAE or directly optimize the enviroments using DSAGE for different domains.

| Config file                                     | Experiment                                |
| ----------------------------------------------- | ----------------------------------------- |
| config/warehouse/even/CMA-MAE_NCA_a=0.gin       | Warehouse (even) CMA-MAE (a=0) + NCA      |
| config/warehouse/even/CMA-MAE_NCA_a=1.gin       | Warehouse (even) CMA-MAE (a=1) + NCA      |
| config/warehouse/even/CMA-MAE_NCA_a=5.gin       | Warehouse (even) CMA-MAE (a=5) + NCA      |
| config/warehouse/even/MAP-Elites_NCA_a=5.gin    | Warehouse (even) MAP-Elites (a=5) + NCA   |
| config/warehouse/even/DSAGE_a=5.gin             | Warehouse (even) DSAGE (a=5)              |
| config/warehouse/uneven/CMA-MAE_NCA_a=0.gin     | Warehouse (uneven) CMA-MAE (a=0) + NCA    |
| config/warehouse/uneven/CMA-MAE_NCA_a=1.gin     | Warehouse (uneven) CMA-MAE (a=1) + NCA    |
| config/warehouse/uneven/CMA-MAE_NCA_a=5.gin     | Warehouse (uneven) CMA-MAE (a=5) + NCA    |
| config/warehouse/uneven/MAP-Elites_NCA_a=5.gin  | Warehouse (uneven) MAP-Elites (a=5) + NCA |
| config/warehouse/uneven/DSAGE_a=5.gin           | Warehouse (uneven) DSAGE (a=5)            |
| config/manufacture/CMA-MAE_NCA_a=5.gin          | Manufacture CMA-MAE (a=5) + NCA           |
| config/manufacture/MAP-Elites_NCA_a=5.gin       | Manufacture MAP-Elites (a=5) + NCA        |
| config/manufacture/DSAGE_a=5.gin                | Manufacture DSAGE (a=5)                   |
| config/maze/entropy_path_len/CMA-MAE_NCA.gin    | Maze (w/ entropy) CMA-MAE + NCA           |
| config/maze/entropy_path_len/MAP-Elites_NCA.gin | Maze (w/ entropy) MAP-Elites + NCA        |
| config/maze/entropy_path_len/DSAGE.gin          | Maze (w/ entropy) DSAGE                   |
| config/maze/n_wall_path_len/CMA-MAE_NCA.gin     | Maze (w/o entropy) CMA-MAE + NCA          |
| config/maze/n_wall_path_len/MAP-Elites_NCA.gin  | Maze (w/o entropy) MAP-Elites + NCA       |
| config/maze/n_wall_path_len/DSAGE.gin           | Maze (w/o entropy) DSAGE                  |

### Optimal Optimized Environments (Manufacture and Warehouse Domains)

We include the optimized and baseline environments of size `S` and `S_eval`.

For size `S`:

| Environment                           | Environment file                                                                                                     |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Human-designed warehouse              | maps/warehouse/human/kiva_large_w_mode.json                                                                          |
| DSAGE-optimized warehouse (even)      | maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_a=5.json       |
| CMA-MAE (a=0) warehouse (even)        | maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy.json                               |
| CMA-MAE (a=1) warehouse (even)        | maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_throughput_hamming_a=1.json        |
| CMA-MAE (a=5) warehouse (even)        | maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_throughput_hamming_a=5.json        |
| DSAGE-optimized warehouse (uneven)    | maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_a=5.json       |
| CMA-MAE (a=0) warehouse (uneven)      | maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven.json                        |
| CMA-MAE (a=1) warehouse (uneven)      | maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=1.json |
| CMA-MAE (a=5) warehouse (uneven)      | maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=5.json |
| Human-designed manufacture            | maps/manufacture/human/manufacture_large_93_stations.json                                                            |
| DSAGE-optimized manufacture           | maps/manufacture/ours/nca_to_show/manufacture_large_200_agents_dsage_opt_alpha=5_sw=5.json                           |
| CMA-MAE (a=5) manufacture, opt        | maps/manufacture/ours/nca_to_show/manufacture_large_200_agents_cma-mae_opt_alpha=5_sw=5.json                         |
| CMA-MAE (a=5) manufacture, comp DSAGE | maps/manufacture/ours/nca_to_show/manufacture_large_200_agents_cma-mae_idx_0=15_opt_alpha=5_sw=5.json                |

For size `S_eval`:

| Environment                      | Environment file                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Human-designed warehouse         | maps/warehouse/human/kiva_xxlarge_w_mode.json                                                                                               |
| CMA-MAE (a=0) warehouse (even)   | maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt.json                               |
| CMA-MAE (a=1) warehouse (even)   | maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=1.json        |
| CMA-MAE (a=5) warehouse (even)   | maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=5.json        |
| CMA-MAE (a=0) warehouse (uneven) | maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_uneven_w.json                      |
| CMA-MAE (a=1) warehouse (uneven) | maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_uneven_throughput-hamming_a=1.json |
| CMA-MAE (a=5) warehouse (uneven) | maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_uneven_throughput-hamming_a=5.json |
| Human-designed manufacture       | maps/manufacture/human/manufacture_xxlarge_1700_stations.json                                                                               |
| CMA-MAE (a=5) manufacture, opt   | maps/manufacture/ours/xxlarge/manufacture_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=5_sw=5_iter=200.json           |

### Generate Environments with Trained NCA

To generate environments with trained NCA stored in the logging directories, run the following:

```
bash scripts/gen_nca_process.sh DOMAIN LOGDIR SEED_ENV_PATH MODE NCA_ITER NCA_ONLY \
    -s SIM_SCORE_WITH -q QUERY
```

- `DOMAIN`: `kiva`, `manufacture`, or `maze`
- `LOGDIR`: path of log directory
- `SEED_ENV_PATH`: path of the initial environment file
- `MODE`: `best` or `idx`. `best` selects the best NCA generator from the archive according to the objective. `idx` selects NCA of specific index from the archive. The queried index is specified from `QUERY` parameter, see below. **Note: only manufacturing and maze domains support `idx` mode**.
- `NCA_ITER`: number of iteration to run the NCA generator
- `NCA_ONLY`: if `True`, only runs NCA, otherwise also runs MILP repair
- `SIM_SCORE_WITH`: optional. Path of an environment. If provided, compute the similarity score between generated environment and the provided environment.
- `QUERY`: optional. Index of the NCA generator to select from the archive under `idx` mode. For example `"50,50"` chooses the elite stored at index `[50, 50]` of the archive.

The generated environment will be stored to a directory named `nca_process_<env_size>_iter=<NCA_ITER>` under `LOGDIR` along with relevant metrics (such as the similarity score).

## Evaluate the Warehouse and Manufacture Environments

### Running Simulations

After getting the environments, we want to evaluate the environments by running simulations. To do so in the provided agent-based simulator, run the following:

```
bash scripts/run_single_sim.sh SIM_CONFIG MAP_FILE AGENT_NUM AGENT_NUM_STEP_SIZE \
    N_EVALS MODE N_SIM N_WORKERS DOMAIN -r RELOAD
```

- `SIM_CONFIG`: gin simulation configuration file, stored under `pure_simulation` directory under `config/<domain>`
- `MAP_FILE`: path of the environment to run the simulations in
- `AGENT_NUM`: number of agent to start with while running simulations
- `AGENT_NUM_STEP_SIZE`: step size of the number of agents to run simulations
- `N_EVALS`: number of evaluations to run, interpreted differently under different modes, see example below
- `MODE`: `inc_agents` or `constant`, see example below
- `N_SIM`: number of simulations to run
- `N_WORKERS`: number of processes to run in parallel
- `DOMAIN`: `kiva`, `manufacture`, or `maze`
- `RELOAD`: optional, log directory to reload an experiment

There are two modes associated with the `MODE` parameter, namely `inc_agents` and `constant`.

**`inc_agents` mode:** the script will run simulations on the provided environment with an increment number of agents. Specifically, starting with `AGENT_NUM`, it increment the number by step size of `AGENT_NUM_STEP_SIZE`, until the number of agents reaches `AGENT_NUM + N_EVALS`. For each number of agents, it runs the simulations `N_SIM` times with seeds from `0` to `N_SIM - 1`. All simulations run in parallel on `N_WORKERS` processes.

For example, the following command:

```
bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR.gin \
    maps/warehouse/human/kiva_large_w_mode.json \
    50 10 301 inc_agents 50 28 kiva
```

runs 50 simulations with 50 to 351 agents, increment in step size of 10, in the environment `maps/warehouse/human/kiva_large_w_mode.json` with simulation config `config/warehouse/pure_simulation/RHCR.gin` in the warehouse domain.

**`constant` mode**: the script will run simulations on the provided environment with a fixed number of agents with random seeds. Specifically, it runs `N_EVALS` simulations with `AGENT_NUM` agents, in parallel on `N_WORKERS` processes. Other parameters will be ignored but they must be given some dummy values for the script to pick up the correct parameter.

For example, the following command:

```
bash scripts/run_single_sim.sh config/manufacture/pure_simulation/RHCR.gin \
    maps/manufacture/human/manufacture_large_93_stations.json \
    200 10 100 constant 50 28 manufacture
```

runs 100 simulations with 200 agents in environment `maps/manufacture/human/manufacture_large_93_stations.json` with simulation config `config/manufacture/pure_simulation/RHCR.gin`.

### Evaluation Logging Directory Manifest

Running the above scripts will generate separate logging directories under `logs`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_<environment-name>`, e.g.
`2020-12-01_15-00-30_environment-1`. Inside each directory are the
following files:

```text
- results # contains the configuration and result of all simulations, stored in json.
- map.json # the environment file
```

### Results

To generate the results shown in the paper, we will reorganize the evaluation logging directories in the following structure:

```text
<eval_exp_name>
|___ Meta Directory of environment 1
     |__ Evaluation Logging Directory1
     |__ Evaluation Logging Directory2
     |__ meta.yaml
|___ Meta Directory of environment 2
     |__ Evaluation Logging Directory1
     |__ Evaluation Logging Directory2
     |__ meta.yaml
```

The evaluation logging directories are obtained by running the aforementioned script. The parent meta directories musted be created to group different evaluation logging directories. A `meta.yaml` file must be created under each meta directorty. An example `meta.yaml` is as follows:

```
algorithm: "RHCR"
map_size: "large"
mode: "w"
map_from: "CMA-MAE + NCA (alpha=0)"
```

where algorithm is the MAPF planner, map_size is the size of the environment (`large` for `S` and `xxlarge` for `S_eval`), mode is the mode of simulation (in this paper it's always `w`, denoting workstation), and `map_from` is the algorithm used to optimize the environment.

Then, the following script can be used to plot throughput vs. number agents and generate numerical results shown in the paper:

```
bash scripts/plot_throughput.sh <eval_exp_name> cross_n_agents
```

In addition, the following script can be used to plot number of finished tasks vs. timesteps:

```
bash scripts/plot_throughput.sh <eval_exp_name> cross_thr_time
```

In our experiments, we use CMA-MAE to train a collection of NCA generators and generate environments of size `S` and `S_eval`, getting the following throughput compared with human-designed and DSAGE optimized environments:

| Domain             | Algorithm                 | Throughput (`S`)   | Throughput (`S_eval`) |
| ------------------ | ------------------------- | ------------------ | --------------------- |
| Warehouse (even)   | CMA-MAE (a=0)             | 6.79 &plusmn; 0.00 | N/A                   |
| Warehouse (even)   | CMA-MAE (a=1)             | 6.73 &plusmn; 0.00 | N/A                   |
| Warehouse (even)   | CMA-MAE (a=5)             | 6.74 &plusmn; 0.00 | 16.01 &plusmn; 0.00   |
| Warehouse (even)   | DSAGE (a=5)               | 6.35 &plusmn; 0.00 | N/A                   |
| Warehouse (even)   | Human                     | N/A                | N/A                   |
| Warehouse (uneven) | CMA-MAE (a=0)             | 6.89 &plusmn; 0.00 | 12.32 &plusmn; 0.01   |
| Warehouse (uneven) | CMA-MAE (a=1)             | 6.70 &plusmn; 0.00 | 11.56 &plusmn; 0.01   |
| Warehouse (uneven) | CMA-MAE (a=5)             | 6.82 &plusmn; 0.00 | 12.03 &plusmn; 0.00   |
| Warehouse (uneven) | DSAGE (a=5)               | 6.40 &plusmn; 0.00 | N/A                   |
| Warehouse (uneven) | Human                     | N/A                | N/A                   |
| Manufacture        | CMA-MAE (a=5, opt)        | 6.82 &plusmn; 0.00 | 23.11 &plusmn; 0.01   |
| Manufacture        | CMA-MAE (a=5, comp DSAGE) | 6.61 &plusmn; 0.00 | N/A                   |
| Manufacture        | DSAGE (a=5)               | 5.61 &plusmn; 0.12 | N/A                   |
| Manufacture        | Human                     | 5.92 &plusmn; 0.00 | N/A                   |

## Scaling RL Policy in Maze Domain

We used a trined ACCEL agent taken from the repository of [DSAGE](https://github.com/icaros-usc/dsage) to run the simulations.

The NCA generated maze environment of size `S_eval`is `maps/maze/ours/nca_to_show/maze_xxlarge_nca_100_50.json`. To run simulations, use the command:

```
bash scripts/run_single_maze_env.sh maps/maze/ours/nca_to_show/maze_xxlarge_nca_100_50.json 100
```

which runs maze simulation in the environment for 100 times.

We compare the performance of the environment in 100 generated baseline environments. To generate these environments, use the command:

```
bash scripts/maze_random_gen.sh maps/maze/ours/nca_to_show/maze_xxlarge_nca_100_50.json 100 0.8 1.2
```

The command generates 100 maze environments the path length from start to goal of which is within 0.8 and 1.2 times the environment in `maps/maze/ours/nca_to_show/maze_xxlarge_nca_100_50.json`. The maps will be stored under a directory under `logs`.

Then, to run the simulations in the baseline environments:

```
bash scripts/run_maze_baseline_envs.sh <baseline_envs> 100 32
```

which runs 100 simulations in each of the baseline maze environments under `<baseline_envs>` with 32 parallel processes and store the results there.

In our experiments, we obtain 22.3% as the average success rate with 3.0% as the standard error. In comparison, our NCA generated maze environment have success rate 93%.


### Troubleshooting

1. If you encounter the following error while running experiments in the Singularity container:

   ```
   container creation failed: mount /proc/self/fd/3->/usr/local/var/singularity/mnt/session/rootfs error: while mounting image`/proc/self/fd/3: failed to find loop device: no loop devices available
   ```

   Please try downgrading/upgrading the Linux kernel version to `5.15.0-67-generic`, as suggested in [this Github issue](https://github.com/sylabs/singularity/issues/1499).


1. On Linux, if you are running anything in the container from external drivers mounted to the home driver (e.g. `/mnt/project`), you need to add `--bind /mnt/project:/mnt/project` to the singularity command in order to bind that external driver also to the container. For example, if you are running an experiment from an external driver, run with:
    ```
    bash scripts/run_local.sh CONFIG SEED NUM_WORKERS -p /mnt/project
    ```
    The `-p` argument helps you add the `--bind` argument to the singularity command in the script.

## License

The code is released under the MIT License, with the following exceptions:

- `env_search/maze/agents/common.py`, `env_search/maze/agents/distributions.py`,
  `env_search/maze/agents/multigrid_network.py`, `env_search/maze/agents/rl_agent.py`, and
  `env_search/maze/envs/` are adapted from the [PAIRED](https://github.com/ucl-dark/paired)
  codebase, which is released under the [Apache-2.0
  License](https://github.com/ucl-dark/paired/blob/master/LICENSE), and the [DCD
  ](https://github.com/facebookresearch/dcd) repo, which is released under the
  [CC BY-NC 4.0 license](https://github.com/facebookresearch/dcd/blob/main/LICENSE).
- `env_search/maze/agents/saved_models/accel_seed_1/model_20000.tar` is the pre-trained
  ACCEL agent used in the experiments and was obtained directly from the
  original authors with their consent.
- `RHCR/` are adapted and modified from [Rolling-Horizon Collision Resolution (RHCR) repo](https://github.com/Jiaoyang-Li/RHCR) under [USC – Research License](https://github.com/Jiaoyang-Li/RHCR/blob/master/license.md).
