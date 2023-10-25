COMP_FILEPATH="$1"
N_GEN="$2"
PATH_LEN_LOW_TOL="$3"
PATH_LEN_UP_TOL="$4"

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/maze/generator/random_generator.py \
        --compare_maze_filepath "$COMP_FILEPATH" \
        --n_gen "$N_GEN" \
        --path_len_low_tol "$PATH_LEN_LOW_TOL" \
        --path_len_up_tol "$PATH_LEN_UP_TOL"