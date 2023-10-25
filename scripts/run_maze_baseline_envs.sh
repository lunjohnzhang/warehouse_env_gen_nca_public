ENV_DIR="$1"
N_EVALS="$2"
N_WORKERS="$3"

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/maze/agents/run_baseline_envs.py \
        --env_dir "$ENV_DIR" \
        --n_evals "$N_EVALS" \
        --n_workers "$N_WORKERS"