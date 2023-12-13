#!/bin/bash

function run_cmd {
    path_mod="./scripts/train:./scripts/eval:/.configs"
    tmux_name="${SPLIT}_${PRIMITIVE}_${SEED}_${CPU}"
    echo "Executing ${PYTHON_CMD} in tmux session ${tmux_name}."
    tmux new-session -d -s "${tmux_name}"
    tmux send-keys -t "${tmux_name}" "export PYTHONPATH=${path_mod}:${PYTHONPATH}" Enter
    tmux send-keys -t "${tmux_name}" "taskset -c ${CPU} ${PYTHON_CMD}" Enter
}

function generate_data {
    TRAINER_CONFIG="configs/pybullet/trainers/datasets/primitive_${SYMBOLIC_ACTION_TYPE}_dataset.yaml"
    args="--config.exp-name ${EXP_NAME}"
    args="${args} --config.trainer-config ${TRAINER_CONFIG}"
    args="${args} --config.split ${SPLIT}"
    args="${args} --config.primitive ${PRIMITIVE}"
    args="${args} --config.symbolic-action-type ${SYMBOLIC_ACTION_TYPE}"
    args="${args} --config.seed ${SEED}"
    args="${args} --config.device ${DEVICE}"
    
    PYTHON_CMD="python generate_primitive_dataset.py ${args}"
    run_cmd
}

function generate_splits {
    N_JOBS_TRAIN=$(echo "scale=1; $N_JOBS*$TRAIN_VALIDATION_SPLIT" | bc)
    N_JOBS_VALIDATION=$(echo "scale=1; $N_JOBS*(1-$TRAIN_VALIDATION_SPLIT)" | bc)
    LC_NUMERIC=C
    N_JOBS_TRAIN=$(printf "%.0f" "$N_JOBS_TRAIN")
    N_JOBS_VALIDATION=$(printf "%.0f" "$N_JOBS_VALIDATION")
    LC_NUMERIC= # Resetting back to the original locale
    echo "N_JOBS_TRAIN: ${N_JOBS_TRAIN}, N_JOBS_VALIDATION: ${N_JOBS_VALIDATION}"
    CPUS=($(seq 0 $(($N_JOBS-1))))
    SEEDS=($(seq $SEED_OFFSET $(($SEED_OFFSET+$N_JOBS-1))))
    TRAIN_IDS=($(seq 0 $(($N_JOBS_TRAIN-1))))
    VALIDATION_IDS=($(seq $N_JOBS_TRAIN $(($N_JOBS-1))))
}

function run_data_generation {
    generate_splits
    SPLIT="train"
    for idx in "${!TRAIN_IDS[@]}"; do
        SEED="${SEEDS[${idx}]}"
        CPU="${CPUS[${idx}]}"
        generate_data
    done
    
    SPLIT="validation"
    for idx in "${!VALIDATION_IDS[@]}"; do
        SEED="${SEEDS[${idx}]}"
        CPU="${CPUS[${idx}]}"
        generate_data
    done
}