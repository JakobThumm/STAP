#!/bin/bash

DOCKER=false
DEVICE="cpu"
EXP_NAME="datasets"
SYMBOLIC_ACTION_TYPE="valid"
PRIMITIVE="pick"
N_JOBS=12
SEED_OFFSET=0
CPU_OFFSET=0
TRAIN_VALIDATION_SPLIT=0.8
user=root

function run_cmd_tmux {
    path_mod="./scripts/train:./scripts/eval:/.configs"
    tmux_name="${SPLIT}_${PRIMITIVE}_${SEED}_${CPU}"
    echo "Executing ${PYTHON_CMD} in tmux session ${tmux_name}."
    tmux new-session -d -s "${tmux_name}"
    tmux send-keys -t "${tmux_name}" "export PYTHONPATH=${path_mod}:${PYTHONPATH}" Enter
    tmux send-keys -t "${tmux_name}" "taskset -c ${CPU} ${PYTHON_CMD}" Enter
}

function run_cmd_docker {
    path_mod="./scripts/train:./scripts/eval:/.configs"
    docker_name="STAP_${USER}_${SPLIT}_${PRIMITIVE}_${SEED}_${CPU}"
    echo "Executing ${PYTHON_CMD} in docker session ${docker_name}."
    if [ "$user" = "root" ]
    then
        docker run -d --rm \
            --name="${docker_name}" \
            --net=host \
            --volume="$(pwd)/models/:/root/models/" \
            --shm-size=10.24gb \
            stap-train/root:v2 "${PYTHON_CMD}"
    elif [ "$user" = "user" ]
    then
        docker run -d --rm \
            --name="${docker_name}" \
            --net=host \
            --volume="$(pwd)/models/:/home/$USER/models/" \
            --shm-size=10.24gb \
            stap-train/$USER:v2 "${PYTHON_CMD}"
    else
    echo "User mode unkown. Please choose user, root, or leave it out for default user"
    fi
    
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
    if [ "$DOCKER" = true ]; then
        run_cmd_docker
    else
        run_cmd_tmux
    fi
}

function generate_splits {
    N_JOBS_TRAIN=$(echo "scale=1; $N_JOBS*$TRAIN_VALIDATION_SPLIT" | bc)
    N_JOBS_VALIDATION=$(echo "scale=1; $N_JOBS*(1-$TRAIN_VALIDATION_SPLIT)" | bc)
    LC_NUMERIC=C
    N_JOBS_TRAIN=$(printf "%.0f" "$N_JOBS_TRAIN")
    N_JOBS_VALIDATION=$(printf "%.0f" "$N_JOBS_VALIDATION")
    LC_NUMERIC= # Resetting back to the original locale
    # If N_JOBS_VALIDATION is 0, we set it to 1 and N_JOBS_TRAIN to N_JOBS-1.
    if [[ $N_JOBS_VALIDATION -eq 0 ]]; then
        N_JOBS_VALIDATION=1
        N_JOBS_TRAIN=$(($N_JOBS-$N_JOBS_VALIDATION))
    fi
    echo "N_JOBS_TRAIN: ${N_JOBS_TRAIN}, N_JOBS_VALIDATION: ${N_JOBS_VALIDATION}"
    CPUS=($(seq $CPU_OFFSET $(($N_JOBS+$CPU_OFFSET-1))))
    SEEDS=($(seq $SEED_OFFSET $(($SEED_OFFSET+$N_JOBS-1))))
    TRAIN_IDS=($(seq 0 $(($N_JOBS_TRAIN-1))))
    VALIDATION_IDS=($(seq $N_JOBS_TRAIN $(($N_JOBS-1))))
}

function run_data_generation {
    generate_splits
    SPLIT="train"
    for train_idx in "${TRAIN_IDS[@]}"; do
        SEED="${SEEDS[${train_idx}]}"
        CPU="${CPUS[${train_idx}]}"
        generate_data
    done
    
    SPLIT="validation"
    for validation_idx in "${VALIDATION_IDS[@]}"; do
        SEED="${SEEDS[${validation_idx}]}"
        CPU="${CPUS[${validation_idx}]}"
        generate_data
    done
}

function generate_full_split() {
    # One generation is one primitive and one symbolic action type.
    # We have to generate N_GENERATIONS = N_Primitives * N_Symbolic_Action_Types generations.
    # We have N_JOBS_PER_GENERATION = N_JOBS/N_GENERATIONS jobs per generation.
    # The SEED_OFFSET should increase by N_JOBS_PER_GENERATION for each generation.
    # The CPU_OFFSET should increase by N_JOBS_PER_GENERATION for each generation.
    # This function sets N_JOBS to N_JOBS_PER_GENERATION and SEED_OFFSETS and CPU_OFFSETS to the correct values.

    # Length of list PRIMITIVES * Length of list SYMBOLIC_ACTION_TYPES
    N_GENERATIONS=$(echo "${#PRIMITIVES[@]} * ${#SYMBOLIC_ACTION_TYPES[@]}" | bc)
    echo "N_GENERATIONS: ${N_GENERATIONS}"
    N_JOBS_PER_GENERATION=$(echo "scale=1; $N_JOBS/$N_GENERATIONS" | bc)
    LC_NUMERIC=C
    N_JOBS_PER_GENERATION=$(printf "%.0f" "$N_JOBS_PER_GENERATION")
    LC_NUMERIC= # Resetting back to the original locale
    echo "N_JOBS_PER_GENERATION: ${N_JOBS_PER_GENERATION}"
    # SEED_OFFSETS=SEED_OFFSET:N_JOBS_PER_GENERATION:SEED_OFFSET+N_JOBS_PER_GENERATION*N_GENERATIONS
    SEED_OFFSETS=($(seq $SEED_OFFSET $N_JOBS_PER_GENERATION $(($SEED_OFFSET+$N_JOBS_PER_GENERATION*$N_GENERATIONS-1))))
    CPU_OFFSETS=($(seq $CPU_OFFSET $N_JOBS_PER_GENERATION $(($CPU_OFFSET+$N_JOBS_PER_GENERATION*$N_GENERATIONS-1))))
    N_JOBS=$N_JOBS_PER_GENERATION
}

# Function to run the existing data generation with different splits and primitives.
function run_variants {
  local counter=0
  generate_full_split
  for action_type in "${SYMBOLIC_ACTION_TYPES[@]}"; do
    for primitive in "${PRIMITIVES[@]}"; do
      SYMBOLIC_ACTION_TYPE=$action_type
      PRIMITIVE=$primitive
      SEED_OFFSET=${SEED_OFFSETS[${counter}]}
      CPU_OFFSET=${CPU_OFFSETS[${counter}]}
      echo "Running data generation for ${PRIMITIVE} with ${SYMBOLIC_ACTION_TYPE} symbolic action type with seed offset ${SEED_OFFSET} and cpu offset ${CPU_OFFSET}."
      run_data_generation
      counter=$((counter+1))
    done
  done
}