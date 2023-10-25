#!/bin/bash

USAGE="Usage: bash scripts/run_single_sim.sh SIM_CONFIG MAP_FILE AGENT_NUM N_EVALS MODE N_SIM N_WORKERS"

SIM_CONFIG="$1"
MAP_FILE="$2"
AGENT_NUM="$3"
AGENT_NUM_STEP_SIZE="$4"
N_EVALS="$5"
MODE="$6"
N_SIM="$7"
N_WORKERS="$8"
DOMAIN="$9"
RELOAD="${10}"

print_header() {
  echo
  echo "------------- $1 -------------"
}

print_header "Create logging directory"
DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="slurm_logs/slurm_${DATE}"
echo "SLURM Log directory: ${LOGDIR}"

SCHEDULER_SCRIPT="${LOGDIR}/scheduler.slurm"
SCHEDULER_OUTPUT="${LOGDIR}/scheduler.out"

mkdir -p "$LOGDIR"

echo "Starting scheduler from: ${SCHEDULER_SCRIPT}"

echo "\
#!/bin/bash
#SBATCH --job-name=single_sim
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=$N_WORKERS
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=48:00:00
#SBATCH --account=nikolaid_548
#SBATCH --output $SCHEDULER_OUTPUT
#SBATCH --error $SCHEDULER_OUTPUT
#SBATCH --partition=epyc-64

echo
echo \"========== Start ==========\"
date

bash scripts/run_single_sim.sh $SIM_CONFIG $MAP_FILE $AGENT_NUM $AGENT_NUM_STEP_SIZE $N_EVALS $MODE $N_SIM $N_WORKERS $DOMAIN $RELOAD

echo
echo \"========== Done ==========\"
date
date" > "$SCHEDULER_SCRIPT"


# Submit the scheduler script.
print_header "Submitting script"
sbatch "$SCHEDULER_SCRIPT"
