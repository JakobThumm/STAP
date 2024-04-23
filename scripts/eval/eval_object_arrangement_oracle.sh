#!/bin/bash

set -e

function run_cmd {
    ${CMD}
}

function eval_planner {
    args=""
    args="${args} --planner-config ${PLANNER_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    if [ ${#SCOD_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --scod-checkpoints ${SCOD_CHECKPOINTS[@]}"
    fi
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    if [[ ! -z "${LOAD_PATH}" ]]; then
        args="${args} --load-path ${LOAD_PATH}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 1"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 1"
    fi
    CMD="python scripts/eval/eval_planners.py ${args}"
    run_cmd
}

function run_planners {
    for task in "${TASKS[@]}"; do
        PLANNER_OUTPUT_PATH="${PLANNER_OUTPUT_ROOT}/${task}"
        ENV_CONFIG="${TASK_ROOT}/${task}.yaml"

        for planner in "${PLANNERS[@]}"; do
            PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

            POLICY_CHECKPOINTS=()
            for primitive in "${PRIMITIVES[@]}"; do
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${CHECKPOINT}.pt")
            done

            SCOD_CHECKPOINTS=()

            if [[ "${planner}" == *_oracle_*dynamics ]]; then
                DYNAMICS_CHECKPOINT=""
            else
                DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CHECKPOINT}.pt"
            fi
    
            eval_planner
        done
    done
}

# Evaluation tasks: Uncomment tasks to evaluate.
TASK_ROOT="configs/pybullet/envs/official/sim_domains"
TASKS=(
    "object_arrangement/oracle_task_0"
    "object_arrangement/oracle_task_1"
    "object_arrangement/oracle_task_2"
    "object_arrangement/oracle_task_3"
    "object_arrangement/oracle_task_4"
    "object_arrangement/oracle_task_5"
    "object_arrangement/oracle_task_6"
    "object_arrangement/oracle_task_7"
    "object_arrangement/oracle_task_8"
    "object_arrangement/oracle_task_9"
    "object_arrangement/oracle_task_10"
    "object_arrangement/oracle_task_11"
    "object_arrangement/oracle_task_12"
)

# Planners: Uncomment planners to evaluate.
PLANNER_CONFIG_PATH="configs/pybullet/planners"
PLANNERS=(
    "policy_cem_arrangement_no_custom_fns"
)

# Setup.
user=${1:-user}

if [ "$user" = "root" ]
    then
    STAP_PATH="/root"
elif [ "$user" = "user" ]
    then
    STAP_PATH="/home/$USER"
else
    echo "User mode unknown. Please choose user, root, or leave out for default user"
fi

input_path="${STAP_PATH}/models"
output_path="${STAP_PATH}/models/eval"

ENV_KWARGS="--gui 0 --closed-loop 1 --use_informed_dynamics 1"

# Evaluate planners.
exp_name="planning"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"
PRIMITIVES=("pick" "place" "static_handover")

CHECKPOINT="final_model"
POLICY_INPUT_PATH="${input_path}/policies_irl"
DYNAMICS_INPUT_PATH="${input_path}/dynamics_irl/pick_place_static_handover_dynamics"
run_planners