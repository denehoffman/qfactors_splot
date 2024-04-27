#!/bin/bash

# Number of iterations to run
NUM_ITERATIONS=10

# Directory where sQ-factors analysis script is located
SCRIPT_DIR="/home/zbaldwin/qfactors_splot"

QUEUE=red

SLEEP=3

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

# Loop through the number of iterations
for (( i=0; i<$NUM_ITERATIONS; i++ ))
do
    # Submit the job to Slurm 
    sbatch --export=ALL,ITERATION=$i,SCRIPT_DIR=$SCRIPT_DIR \
           --job-name="sQ_factor_iter_$i" \
           --ntasks=${THREADS} \
           --partition=${QUEUE} \
           --mem=${MEM} \
           --time=02:00:00 \
           --output="${SCRIPT_DIR}/slurm_results/qfactor_splot_iter_${i}_%j.out" \
           --error="${SCRIPT_DIR}/slurm_results/qfactor_splot_iter_${i}_%j.err" \
           "${SCRIPT_DIR}/run_analysis.sh" $i
    sleep $SLEEP
done
