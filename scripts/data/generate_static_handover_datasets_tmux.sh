#!/bin/bash
set -e

source scripts/data/helper_functions.sh
N_JOBS=40
# Experiments.
DEVICE="cpu"
EXP_NAME="datasets"
SEED_OFFSET=0
CPU_OFFSET=0
TRAIN_VALIDATION_SPLIT=0.8
PRIMITIVES=("static_handover")
SYMBOLIC_ACTION_TYPES=("valid" "invalid")

# Call the function to start the process.
run_variants
