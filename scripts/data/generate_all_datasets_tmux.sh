#!/bin/bash
N_JOBS=$1
set -e

function run_cmd {
    echo ""
    path_mod="./scripts/train:./scripts/eval:/.configs"
    tmux_name="${SPLIT}_${PRIMITIVE}_${SEED}_${CPU}"

    tmux new-session -d -s "${tmux_name}"
    tmux send-keys -t "${tmux_name}" "export PYTHONPATH=${path_mod}:${PYTHONPATH}" Enter
    tmux send-keys -t "${tmux_name}" "taskset -c ${CPU} ${PYTHON_CMD}" Enter
}

function generate_data {
    args="--config.exp-name ${EXP_NAME}"
    args="${args} --config.split ${SPLIT}"
    args="${args} --config.primitive ${PRIMITIVE}"
    args="${args} --config.symbolic-action-type ${SYMBOLIC_ACTION_TYPE}"
    args="${args} --config.seed ${SEED}"
    
    PYTHON_CMD="python generate_primitive_dataset.py ${args}"
    run_cmd
}

function run_data_generation {
    for idx in "${!SEEDS[@]}"; do
        SEED="${SEEDS[${idx}]}"
        CPU="${CPUS[${idx}]}"
        generate_data
    done
}

# Experiments.
EXP_NAME="datasets"
# Sequence from 0 to N_JOBS-1.
SEEDS=($(seq 0 $(($N_JOBS-1))))
CPUS=($(seq 0 $(($N_JOBS-1))))

# Function to run the existing data generation with different splits and primitives.
function run_variants {
  local splits=("train" "validation")
  local primitives=("pick" "place" "static_handover")
  local action_types=("valid" "invalid")
  for action_type in "${action_types[@]}"; do
    for split in "${splits[@]}"; do
        for primitive in "${primitives[@]}"; do
            SYMBOLIC_ACTION_TYPE=$action_type
            SPLIT=$split
            PRIMITIVE=$primitive
            run_data_generation
        done
    done
  done
}

# Call the function to start the process.
run_variants
