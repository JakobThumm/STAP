#!/bin/bash

source scripts/data/helper_functions.sh

set -e

# Experiments.
DEVICE="cpu"
EXP_NAME="datasets"
SYMBOLIC_ACTION_TYPE="valid"
PRIMITIVE="pick"
N_JOBS=12
SEED_OFFSET=0
CPU_OFFSET=0
TRAIN_VALIDATION_SPLIT=0.8

run_data_generation

# Experiments.
SYMBOLIC_ACTION_TYPE="invalid"
N_JOBS=12
SEED_OFFSET=12
CPU_OFFSET=12
TRAIN_VALIDATION_SPLIT=0.8

run_data_generation