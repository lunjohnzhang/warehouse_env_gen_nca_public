#!/bin/bash

USAGE="Usage: bash scripts/plot_heatmap.sh LOGDIR MODE QUERY VIDEO"

LOGDIR="$1"
MODE="$2"
QUERY="$3"
VIDEO="$4"

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${VIDEO}" ]
then
  VIDEO="False"
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/maze_viz.py \
        --logdir "$LOGDIR" \
        --mode "$MODE" \
        --query "$QUERY" \
        --video "$VIDEO"