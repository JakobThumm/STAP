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

function train_value {
    args=""
    args="${args} --trainer-config ${TRAINER_CONFIG}"
    args="${args} --agent-config ${AGENT_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ! -z "${ENCODER_CHECKPOINT}" ]; then
        args="${args} --encoder-checkpoint ${ENCODER_CHECKPOINT}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    args="${args} --train-data-checkpoints ${TRAIN_DATA_CHECKPOINTS}"
    args="${args} --eval-data-checkpoints ${EVAL_DATA_CHECKPOINTS}"
    if [ ! -z "${NAME}" ]; then
        args="${args} --name ${NAME}"
    fi

    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --path ${VALUE_OUTPUT_PATH}_debug"
        args="${args} --overwrite"
        args="${args} --num-train-steps 10"
        args="${args} --num-eval-episodes 10"
    else
        args="${args} --path ${VALUE_OUTPUT_PATH}"
    fi

    CMD="python scripts/train/train_agent.py ${args}"
    run_cmd
}

function run_value {
    ENV_CONFIG="configs/pybullet/envs/official/primitives/heavy/${PRIMITIVE}.yaml"

    TRAIN_DATA_CHECKPOINTS=""
    for seed in "${TRAIN_SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/train_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        TRAIN_DATA_CHECKPOINTS="${TRAIN_DATA_CHECKPOINTS} ${data_path}"
    done

    EVAL_DATA_CHECKPOINTS=""
    for seed in "${VALIDATION_SEEDS[@]}"; do
        data_path="${DATA_CHECKPOINT_PATH}/validation_${SYMBOLIC_ACTION_TYPE}_${PRIMITIVE}_${seed}/train_data"
        EVAL_DATA_CHECKPOINTS="${EVAL_DATA_CHECKPOINTS} ${data_path}"
    done

    train_value
}

# Setup.
SBATCH_SLURM="scripts/train/train_juno.sh"
DEBUG=0
user=${1:-user}
gpu=${2:-cpu}

input_path="models"
output_path="models"

ENV_KWARGS="--gui 0"

# Train critic library.
exp_name="value_fns_irl"
VALUE_OUTPUT_PATH="${output_path}/${exp_name}"

SYMBOLIC_ACTION_TYPE="valid"
TRAIN_SEEDS=($(seq 0 15))
VALIDATION_SEEDS=($(seq 16 19))
DATA_CHECKPOINT_PATH="${input_path}/datasets"

# Details: 1M episodes, Logistic Regression loss for Q-networks, ensemble of 8 Q-networks.
TRAINER_CONFIG="configs/pybullet/trainers/value/value_iter-3M.yaml"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/value/sac_ens_value_mse.yaml"
PRIMITIVE="pick"
run_value

# Details: 1M episodes, MSE loss for Q-networks, ensemble of 8 Q-networks.
TRAINER_CONFIG="configs/pybullet/trainers/value/value_iter-3M.yaml"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/value/sac_ens_value_mse.yaml"
PRIMITIVE="place"
run_value

# Details: 1M episodes, MSE loss for Q-networks, ensemble of 8 Q-networks.
TRAINER_CONFIG="configs/pybullet/trainers/value/value_iter-3M.yaml"
AGENT_CONFIG="configs/pybullet/agents/multi_stage/value/sac_ens_value_mse.yaml"
PRIMITIVE="static_handover"
run_value
