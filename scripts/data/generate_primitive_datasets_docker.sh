#!/bin/bash

source scripts/data/helper_functions.sh

set -e

# Experiments.
DOCKER=true
DEVICE="cpu"
EXP_NAME="datasets"
SYMBOLIC_ACTION_TYPE="valid"
PRIMITIVE="pick"
N_JOBS=12
SEED_OFFSET=0
CPU_OFFSET=0
TRAIN_VALIDATION_SPLIT=0.8

run_data_generation