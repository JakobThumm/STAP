#!/bin/bash

set -e

function run_cmd {
    docker_command="docker run -d --rm "
    options="--net=host --shm-size=10.24gb"
    image="stap-train"

    if [ "$gpu" = "gpu" ]
    then
        options="${options} --gpus all"
        image="${image}-gpu"
    fi

    if [ "$user" = "root" ]
        then
        options="${options} --volume="$(pwd)/models/:/root/models/""
        image="${image}/root:v2"
    elif [ "$user" = "user" ]
        then
        options="${options} --volume="$(pwd)/models/:/home/$USER/models/" --user=$USER"
        image="${image}/$USER:v2"
    else
        echo "User mode unknown. Please choose user, root, or leave out for default user"
    fi

    echo "Running docker command: ${docker_command} ${options} ${image} ${CMD}"

    ${docker_command} \
        ${options} \
        ${image} \
        ${CMD}
}

function train_dynamics {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --dynamics-config ${DYNAMICS_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"

    if [ ! -z "${NAME}" ]; then
        args="${args} --name ${NAME}"
    fi
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10000"
    else
        args="${args} --path ${DYNAMICS_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_dynamics.py ${args}"
    run_cmd
}

function run_dynamics {
    NAME=""
    POLICY_CHECKPOINTS=()
    for primitive in "${PRIMITIVES[@]}"; do
        POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${POLICY_CHECKPOINT}.pt")

        if [ -z "${NAME}" ]; then
            NAME="${primitive}"
        else
            NAME="${NAME}_${primitive}"
        fi
    done
    NAME="${NAME}_dynamics"

    train_dynamics
}

# Setup.
DEBUG=0
user=${1:-user}
gpu=${2:-cpu}

input_path="${STAP_PATH}/models"
output_path="${STAP_PATH}/models"
ENV_KWARGS="--gui 0"


# Train dynamics.
DYNAMICS_CONFIG="${STAP_PATH}/configs/pybullet/dynamics/table_env.yaml"
TRAINER_CONFIG="${STAP_PATH}/configs/pybullet/trainers/dynamics/dynamics_iter-0.75M.yaml"
PRIMITIVES=("pick" "place" "static_handover")

# Uncomment to train dynamics for RL agents (scripts/train/train_agents.sh).
# exp_name="dynamics_rl"
# DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}"
# POLICY_INPUT_PATH="${input_path}/agents_rl"
# POLICY_CHECKPOINT="final_model"
# run_dynamics

# Uncomment to train dynamics for inverse RL policies (scripts/train/train_policies.sh).
exp_name="dynamics_irl"
DYNAMICS_OUTPUT_PATH="${output_path}/${exp_name}"
POLICY_INPUT_PATH="${input_path}/policies_irl"
POLICY_CHECKPOINT="final_model"
run_dynamics