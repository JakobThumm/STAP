# Train values
python scripts/train/train_agent.py --trainer-config configs/pybullet/trainers/value/value_iter-1K.yaml \
        --agent-config configs/pybullet/agents/multi_stage/value/sac_ens_value_logistics.yaml \
        --env-config configs/pybullet/envs/official/primitives/heavy/pick.yaml \
        --seed 0 \
        --train-data-checkpoints models/datasets/train_valid_pick_0/train_data \
        --eval-data-checkpoints models/datasets/validation_valid_pick_16/train_data \
        --path models/value_fns_irl \
        --overwrite \
        --gui 0
# Train values full path
python scripts/train/train_agent.py --trainer-config /home/thummj/configs/pybullet/trainers/value/value_iter-1K.yaml \
        --agent-config /home/thummj/configs/pybullet/agents/multi_stage/value/sac_ens_value_logistics.yaml \
        --env-config /home/thummj/configs/pybullet/envs/official/primitives/heavy/pick.yaml \
        --seed 0 \
        --train-data-checkpoints /home/thummj/models/datasets/train_valid_pick_0/train_data \
        --eval-data-checkpoints /home/thummj/models/datasets/train_valid_pick_0/train_data \
        --path /home/thummj/models/value_fns_irl \
        --overwrite \
        --gui 0
# Train policies
python scripts/train/train_agent.py  \
        --trainer-config configs/pybullet/trainers/policy/policy-10K.yaml \
        --agent-config configs/pybullet/agents/multi_stage/policy/sac_policy.yaml \
        --env-config configs/pybullet/envs/official/primitives/heavy/pick_eval.yaml \
        --critic-checkpoint models/value_fns_irl/pick/final_model.pt \
        --seed 0 \
        --train-data-checkpoints models/datasets/train_valid_pick_0/train_data \
        --eval-data-checkpoints models/datasets/validation_valid_pick_16/train_data \
        --overwrite \
        --path models/policies_irl \
        --eval-recording-path plots/policies_irl \
        --gui 0

# Train dynamics
python scripts/train/train_dynamics.py  \
    --trainer-config configs/pybullet/trainers/dynamics/dynamics_iter-10K.yaml \
    --dynamics-config configs/pybullet/dynamics/table_env.yaml \
    --policy-checkpoints models/policies_irl/pick/final_model.pt \
    --seed 0  \
    --name pick_dynamics \
    --path models/dynamics_rl \
    --gui 0