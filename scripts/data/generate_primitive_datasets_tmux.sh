#!/bin/bash

source scripts/data/helper_functions.sh

set -e

PRIMITIVE=${1:-pick}

# Experiments.
DEVICE="cpu"
EXP_NAME="datasets"
SYMBOLIC_ACTION_TYPE="valid"
PRIMITIVE="pick"
N_JOBS=20
if [ "$PRIMITIVE" = "pick" ]
then
    SEED_OFFSET=0
elif [ "$PRIMITIVE" = "place" ]
then
    SEED_OFFSET=20
elif [ "$PRIMITIVE" = "static_handover" ]
then
    SEED_OFFSET=40
else
    echo "Primitive unknown. Please choose pick or place"
fi
CPU_OFFSET=0
TRAIN_VALIDATION_SPLIT=0.8

run_data_generation

# Experiments.
SYMBOLIC_ACTION_TYPE="invalid"
N_JOBS=20
if [ "$PRIMITIVE" = "pick" ]
then
    SEED_OFFSET=60
elif [ "$PRIMITIVE" = "place" ]
then
    SEED_OFFSET=80
elif [ "$PRIMITIVE" = "static_handover" ]
then
    SEED_OFFSET=60
else
    echo "Primitive unknown. Please choose pick or place"
fi
CPU_OFFSET=20
TRAIN_VALIDATION_SPLIT=0.8

run_data_generation