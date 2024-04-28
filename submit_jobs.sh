#!/bin/bash

# Current username
USERNAME=$(whoami)

# Directory where sQ-factors analysis script is located
SCRIPT_DIR="/home/zbaldwin/qfactors_splot"

DEFAULT_QUEUE="red"
DEFAULT_NUM_ITERATIONS=3     

RUN_PARTITION_UPDATE=false

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --queue <queue>             Specify the Slurm queue (default: $DEFAULT_QUEUE)"
    echo "  --num-iter <count>          Specify the number of iterations (default: $DEFAULT_NUM_ITERATIONS)"
    echo "  --update-partition          Update the partition for pending jobs"
    echo "  --help                      Display this help message"
    exit 1
}

if [[ "$1" == "--help" ]]; then
    usage 
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --queue)
            QUEUE="$2"
            shift
            ;;
            --num-iter)
            NUM_ITERATIONS="$2"
            shift
            ;; 
            --update-partition)
            RUN_PARTITION_UPDATE=true
            ;;
        *)  # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Use default queue and iterations if no input provided
QUEUE=${QUEUE:-$DEFAULT_QUEUE}
NUM_ITERATIONS=${NUM_ITERATIONS:-$DEFAULT_NUM_ITERATIONS}

# Usage for CMU MEG cluster
if [ "$QUEUE" == "green" ]; then
  CPUMEM=1590
  THREADS=40
elif [ "$QUEUE" == "red" ]; then
  CPUMEM=1990
  THREADS=32
elif [ "$QUEUE" == "blue" ]; then
  CPUMEM=990
  THREADS=64
else
  CPUMEM=990
fi

let MEM=$THREADS*CPUMEM
SLEEP=1

# Update partition if specified
if $RUN_PARTITION_UPDATE; then
    read -p 'Which partition do you want to switch to? ': partvar
    for i in $(squeue -u $USERNAME -h -t PD -o %i); do
        scontrol update jobid=$i partition=$partvar
    done
    exit 
fi


# Loop through the number of iterations
for (( i=1; i<($NUM_ITERATIONS+1); i++ ))
do
    LOG_DIR="${SCRIPT_DIR}/log/${i}"    
    mkdir -p "$LOG_DIR"      

    # Submit the job to Slurm 
    sbatch --export=ALL,ITERATION=$i,SCRIPT_DIR=$SCRIPT_DIR,LOG_DIR=$LOG_DIR \
           --job-name="sQ_factor_iter_$i" \
           --ntasks=${THREADS} \
           --partition=${QUEUE} \
           --mem=${MEM} \
           --time=02:00:00 \
           --output="${LOG_DIR}/qfactor_splot_iter_${i}.out" \
           --error="${LOG_DIR}/qfactor_splot_iter_${i}.err" \
           "${SCRIPT_DIR}/run_analysis.sh" $i
    #sleep $SLEEP
done
