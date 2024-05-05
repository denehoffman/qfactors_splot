#!/bin/bash

echo -------------------SLURM JOB INFO-----------------------------
echo 'Submitted from' $SLURM_SUBMIT_HOST
echo 'on' $SLURM_JOB_PARTITION 'queue'
echo 'Job identifier is' $SLURM_JOB_ID
echo 'Job name is' $SLURM_JOB_NAME
echo 'The username is' $SLURM_JOB_USER
echo -n 'Job is running on node '; echo $SLURM_JOB_NODELIST
echo --------------------------------------------------------------

ITERATION=$1

OUTPUT_DIR="${SCRIPT_DIR}/slurm"                                                                                                                                                                      
mkdir -p "$OUTPUT_DIR"

# Run the Python script with the current iteration
python3.11 "$SCRIPT_DIR/analysis" --num-sig 1000 --num-bkg 1000 --num-iter "$ITERATION" --output "$OUTPUT_DIR" 

