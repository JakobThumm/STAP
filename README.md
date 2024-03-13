# STAP: Sequencing Task-Agnostic Policies
The official code repository for *"STAP: Sequencing Task-Agnostic Policies,"* presented at ICRA 2023. 
For a brief overview of our work, please refer to our [project page](https://sites.google.com/stanford.edu/stap).
Further details can be found in our paper available on [arXiv](https://arxiv.org/abs/2210.12250).

<img src="readme/stap-preview.png" alt="STAP Preview"/>


## Overview

The STAP framework can be broken down into two phases: (1) train skills offline (i.e. policies, Q-functions, dynamics models, uncertainty quantifers); 2) plan with skills online (i.e. motion planning, task and motion planning). 
We provide implementations for both phases:

### :hammer_and_wrench: Train Skills Offline
- **Skill library:** A suite of reinforcement learning (RL) and inverse RL algorithms to learn four skills: `Pick`, `Place`, `Push`, `Pull`. 
- **Dynamics models:** Trainers for learning skill-specific dynamics models from off-policy transition experience.
- **UQ models:** Sketching Curvature for Out-of-Distribution Detection ([SCOD](https://arxiv.org/abs/2102.12567)) implementation and trainers for Q-network epistemic uncertainty quantification (UQ).

### :rocket: Plan with Skills Online
- **Motion planners (STAP):** A set of sampling-based motion planners including randomized sampling, cross-entropy method, planning with uncertainty-aware metrics, and combinations.
- **Task and motion planners (TAMP):** Coupling PDDL-based task planning with STAP-based motion planning.

### :card_index_dividers:	Additionals
- **Baseline methods:** Implementations of Deep Affordance Foresight ([DAF](https://arxiv.org/abs/2011.08424)) and parameterized-action [Dreamer](https://arxiv.org/abs/1912.01603).
- **3D Environments:** PyBullet tabletop manipulation environment with domain randomization.

## Setup

### System Requirements
This repository is primarily tested on Ubuntu 20.04 and macOS Monterey with Python 3.8.10.

### Installation
Python packages are managed through Pipenv.
Follow the installation procedure below to get setup:

```bash
# Install pyenv.
curl https://pyenv.run | bash 
exec $SHELL          # Restart shell for path changes to take effect.
pyenv install 3.8.10 # Install a Python version.
pyenv global 3.8.10  # Set this Python to default.

# Clone repository.
git clone https://github.com/agiachris/STAP.git --recurse-submodules
cd STAP

# Install pipenv.
pip install pipenv
pipenv install --dev
pipenv sync
```

Use `pipenv shell` The load the virtual environment in the current shell.

## Instructions

### Basic Usage
STAP supports [training skills](#training-skills), [dynamics models](#training-dynamics), and composing these components at test-time for [planning](#evaluating-planning).
- **STAP module:** The majority of the project code is located in the package `stap/`. 
- **Scripts:** Code for launching experiments, debugging, plotting, and visualization is under `scripts/`.
- **Configs:** Training and evaluation functionality is determined by `.yaml` configuration files located in `configs/`.

#### Launch Scripts
We provide launch scripts for training STAP's required models below.
The launch scripts also support parallelization on a cluster managed by SLURM, and will otherwise default to sequentially processing jobs.


### Model Checkpoints
As an alternative to training skills and dynamics models from scratch, we provide checkpoints that can be downloaded and directly used to [evaluate STAP planners](#evaluating-planning).
Run the following commands to download the model checkpoints to the default `models/` directory (this requires ~10GBs of disk space):
```bash
pipenv shell  # script requires gdown
bash scripts/download/download_checkpoints.sh
```
Once the download has finished, the `models/` directory will contain: 
- Skills trained with [RL](#reinforcement-learning) (`agents_rl`) and their dynamics models (`dynamics_rl`)
- Skills trained with [inverse RL](#inverse-reinforcement-learning) (`policies_irl`) and their dynamics models (`dynamics_irl`)
- Demonstration data used to train inverse RL skills (`datasets`)
- Checkpoints for the [Deep Affordance Foresight](#baseline-deep-affordance-foresight) baseline (`baselines`)


#### Checkpoint Results
We also provide the planning results that correspond to [evaluating STAP](#stap-for-motion-planning) on [these checkpoints](#model-checkpoints).
To download the results to the default `plots/` directory, run the following command (this requires ~3.5GBs of disk space):
```bash
pipenv shell  # script requires gdown
bash scripts/download/download_results.sh
```

The planning results can be visualized by running `bash scripts/visualize/generate_figures.sh` which will save the figure shown below to `plots/planning-result.jpg`.

<img src="readme/planning-result.jpg" alt="STAP Motion Planning Result"/>


### Training Skills
Skills in STAP are trained independently in custom environments.
We provide two pipelines, [RL](#reinforcement-learning) and [inverse RL](#inverse-reinforcement-learning), for training skills. 
While we use RL in the paper, skills learned via inverse RL yield significantly higher planning performance. 
This is because inverse RL offers more control over the training pipeline, allowing us to tune hyperparameters for data generation and skill training.
We have only tested [SCOD UQ](#optional-uncertainty-quantification) with skills learned via RL.

#### Reinforcement Learning
To simultaneously learn an actor-critic per skill with RL, the relevant command is:
```bash
bash scripts/train/train_agents.sh
```
When the skills have finished training, copy and rename the desired checkpoints.
```bash
python scripts/debug/select_checkpoints.py --clone-name official --clone-dynamics True
```
These copied checkpoints will be used for [planning](#evaluating-planning).

##### (Optional) Uncertainty Quantification
Training SCOD is only required if the skills are intended to be used with an uncertainty-aware planner.
```bash
bash scripts/train/train_scod.sh
```

#### Inverse Reinforcement Learning
To instead use inverse RL to learn a critic, then an actor, we first generate a dataset of demos per skill:
```bash
bash scripts/data/generate_primitive_datasets.sh    # generate skill data
bash scripts/train/train_values.sh                  # train skill critics
bash scripts/train/train_policies.sh                # train skill actors
```


### Training Dynamics
Once the skills have been learned, we can train a dynamics model with:
```bash
bash scripts/train/train_dynamics.sh
```

### Evaluating Planning
With skills and dynamics models, we have all the essential pieces required to solve long-horizon manipulation problems with STAP. 

#### STAP for Motion Planning
To evaluate the motion planners at specified agent checkpoints:
```bash
bash scripts/eval/eval_planners.sh
```
To evaluate variants of STAP, or test STAP on a subset of the 9 evaluation tasks, minor edits can be made to the above launch file.

#### STAP for Task and Motion Planning
To evaluate TAMP involving a PDDL task planner and STAP at specified agent checkpoints:
```bash
bash scripts/eval/eval_tamp.sh
```

### Baseline: Deep Affordance Foresight
Our main baseline is Deep Affordance Foresight (DAF). 
DAF trains a new set of skills for each task, in contrast to STAP which trains a set of skills that are used for all downstream tasks.
DAF is also evaluated on the task it is trained on, whereas STAP must generalize to each new task it is evaluated on.

To train a DAF model on each of the 9 evaluation tasks:
```bash
bash scripts/train/train_baselines.sh
```
When the models have finished training, evaluate them with:
```bash
bash scripts/eval/eval_daf.sh
```

## Current server workflow
```
./build_docker_train.sh user gpu
```
### Generate data
```
./run_docker_train.sh user gpu
./scripts/data/generate_all_datasets_tmux.sh 120
```
Remove docker container.
### Train Q-Functions
Not inside docker: The lower command is a fast version for easy testing.
```
./scripts/train/train_values_docker.sh user gpu
./scripts/train/train_values_docker_fast.sh user gpu
```
### Train Policies
```
./scripts/train/train_policies_docker.sh user gpu
./scripts/train/train_policies_docker_fast.sh user gpu
```
### Train Dynamics
```
./scripts/train/train_dynamics_docker.sh user gpu
./scripts/train/train_dynamics_docker_fast.sh user gpu
```

## Real-world tests
1. Build docker
```
./build_docker_ros.sh user
```
2. Run docker
```
./run_docker_ros.sh user
```
We recommend running `tmux` in the docker container to have multiple tabs ([see controls here](https://www.seanh.cc/2020/12/30/how-to-make-tmux%27s-windows-behave-like-browser-tabs/))
3. Build catkin workspace
```
catkin build
```
4. Export ROS master URI
```
export ROS_MASTER_URI=http://[ROS_MASTER_IP]:11311
```
e.g., 
```
export ROS_MASTER_URI=http://[LOCAL_IP]:11311
```
5. Run example:
    ```
    python scripts/debug/debug_pybullet.py -e configs/pybullet/envs/official/real_domains/debug/real_human_template_env.yaml
    ```
    No custom functions handover:
    ```
    python scripts/eval/eval_planners.py  --planner-config configs/pybullet/planners/policy_cem_screwdriver_no_custom_fns.yaml --env-config configs/pybullet/envs/official/real_domains/screwdriver_handover/task0.yaml --policy-checkpoints models/policies_irl/pick/final_model.pt models/policies_irl/place/final_model.pt models/policies_irl/static_handover/final_model.pt --dynamics-checkpoint models/dynamics_irl/pick_place_static_handover_dynamics/final_model.pt --seed 0 --gui 1 --closed-loop 1 --num-eval 1 --path plots/planning/screwdriver_handover/task0 --verbose 1          
    ```
    Custom functions handover:
    ```
    python scripts/eval/eval_planners.py  --planner-config configs/pybullet/planners/policy_cem_screwdriver_custom_fns.yaml --env-config configs/pybullet/envs/official/real_domains/screwdriver_handover/task0.yaml --policy-checkpoints models/policies_irl/pick/final_model.pt models/policies_irl/place/final_model.pt models/policies_irl/static_handover/final_model.pt --dynamics-checkpoint models/dynamics_irl/pick_place_static_handover_dynamics/final_model.pt --seed 0 --gui 1 --closed-loop 1 --num-eval 1 --path plots/planning/screwdriver_handover/task0 --verbose 1                 
    ```
6. Tmux commands:
    - Open a new tab with `Ctrl`+`T`
    - Close the current tab with `Ctrl`+`Alt`+`W`
    - Switch to a different tab with `Ctrl`+`Page up` (or down)

### Troubleshooting
If the connection between machines is incorrect, check for `ROS_MASTER_URI`, `ROS_IP`, and `ROS_HOSTNAME`.
Make sure you allow communication through the firewall:
E.g. on machine 10.42.0.1 to ROS Master on 10.42.0.69:
```
sudo ufw allow out to 10.42.0.69
sudo ufw allow from 10.42.0.69
```
A great tutorial can be found [here](https://roscon.ros.org/2013/wp-content/uploads/2013/06/Networking-for-ROS-Users.pdf).
[This](http://wiki.ros.org/ROS/NetworkSetup) is also helpful.
And more details concerning connecting ROS within docker containers [here](https://www.finnrietz.dev/linux/ros-docker/).
---
## Citation
Sequencing Task-Agnostic Policies is offered under the [MIT License](https://github.com/agiachris/STAP/blob/main/LICENSE) agreement. 
If you find STAP useful, please consider citing our work:
```
@article{agia2022taps,
  title={STAP: Sequencing Task-Agnostic Policies},
  author={Agia, Christopher and Migimatsu, Toki and Wu, Jiajun and Bohg, Jeannette},
  journal={arXiv preprint arXiv:2210.12250},
  year={2022}
}
```
