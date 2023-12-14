#!/bin/bash
set -e

source scripts/data/helper_functions.sh
N_JOBS=$1

# Experiments.
DOCKER=true
DEVICE="cpu"
EXP_NAME="datasets"
SEED_OFFSET=0
CPU_OFFSET=0
TRAIN_VALIDATION_SPLIT=0.8
PRIMITIVES=("pick" "place" "static_handover")
SYMBOLIC_ACTION_TYPES=("valid" "invalid")

# Call the function to start the process.
run_variants
