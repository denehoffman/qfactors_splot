#!/bin/bash

# Passed argument from the main submit script
ITERATION=$1

SCRIPT_DIR="/home/zbaldwin/qfactors_splot"

# Run the Python script with the current iteration
python3.11 "$SCRIPT_DIR/analysis" --num-sig 100 --num-bkg 100 --num-iter "$ITERATION"

