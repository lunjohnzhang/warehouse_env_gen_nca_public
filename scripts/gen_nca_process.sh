#!/bin/bash

USAGE="Usage: bash scripts/gen_nca_process.sh DOMAIN LOGDIR SEED_ENV_PATH MODE NCA_ITER NCA_ONLY -s SIM_SCORE_WITH -q QUERY"

DOMAIN="$1"
LOGDIR="$2"
SEED_ENV_PATH="$3"
MODE="$4"
NCA_ITER="$5"
NCA_ONLY="$6"

shift 6
while getopts "p:s:q:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      s) SIM_SCORE_WITH=$OPTARG;;
      q) QUERY=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done


if [ -z "${DOMAIN}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${SEED_ENV_PATH}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${NCA_ITER}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${NCA_ONLY}" ]
then
  echo "${USAGE}"
  exit 1
fi


if [ -z "${SIM_SCORE_WITH}" ]
then
  SIM_SCORE_WITH=""
fi


if [ -z "${QUERY}" ]
then
  QUERY=""
fi



SINGULARITY_OPTS="--cleanenv --nv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
if [ "${DOMAIN}" = "kiva" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/warehouse/generator/offline_run_nca_generator.py \
            --logdir "$LOGDIR"\
            --seed_env_path "$SEED_ENV_PATH" \
            --nca-iter "$NCA_ITER" \
            --mode "$MODE" \
            --nca-only "$NCA_ONLY" \
            --sim_score_with "$SIM_SCORE_WITH"
fi

if [ "${DOMAIN}" = "manufacture" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/manufacture/generator/offline_run_nca_generator.py \
            --logdir "$LOGDIR"\
            --seed_env_path "$SEED_ENV_PATH" \
            --nca-iter "$NCA_ITER" \
            --mode "$MODE" \
            --nca-only "$NCA_ONLY" \
            --query "$QUERY" \
            --sim_score_with "$SIM_SCORE_WITH"
fi

if [ "${DOMAIN}" = "maze" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/maze/generator/offline_run_nca_generator.py \
            --logdir "$LOGDIR"\
            --seed_env_path "$SEED_ENV_PATH" \
            --nca-iter "$NCA_ITER" \
            --mode "$MODE" \
            --query "$QUERY"
fi