LEVEL_PATH="$1"
N_EVALS="$2"

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/maze/agents/run_single_rl_agent.py \
        --level_filepath "$LEVEL_PATH" \
        --n_evals "$N_EVALS"