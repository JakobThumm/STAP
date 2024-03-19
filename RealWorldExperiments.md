# Real World Experiment Setup and Steps
This document describes all steps necessary to run the real-world experiments.

## General Setup
### Setting up the workstations
 1. Boot the lab PC with real-time kernel activated: choose in the boot menu `Advanced Options` > `RC-20`  (second from the bottom).
 2. Turn on the control PC (black box on the desk) at the poser switch in the back.
 3. Connectivity -> see the `ethernet_setup.txt` file on the desktop or in this repo:
    a. [OPTIONAL] For internet access connect your laptop to the lab PC and activate Wifi forwarding.
    b. In the settings go to `Network` and make sure that `Franka` is activated for `Ethernet (enp13s0f0)` and `VRPN OptiTrack` is activated for `Ethernet (enp13s0f1)`.
 4. On the OptiTrack PC make sure the human hands, shoulders, torso, clav and head, as well as the screwdriver are activated in OptiTrack Motion. VRPN streaming should already be activated. By default, this should be active.
 5. Open a console and 
    ```
    cd sara_shield_ros
    catkin build
    source devel/setup.bash
    ```
 6. Open google chrome and go to `172.16.0.2/desk` or click on the first bookmark.
    a. It might be neccessary to activate advanced security.
    b. The robot on the right should be yellow with joints locked. If not, press the black E-stop button on the desk.
    c. Open the robot brakes with the open lock symbol above the robot symbol. Approve with `Open`.
    d. The robot should be white now. Open the black E-stop button on the desk to go to blue mode. The robot now takes commands.
    e. PRACTICE: Press the black E-stop button on the desk to go to white mode again. This is your emergency stop! 
    f. When the robot is in white mode, you can press the two black buttons on the end effector to move the robot to a custom position. This is useful if the robot got stuck in a pose that's out of joint limits.
    g. Go back to blue mode again.

### Setting up your local workstation
 1. In the settings go to `Network` and create a Wired connection with IPv4 address in the same subnet as the `Manual` connection on the lab PC. (Optionally, activate Wifi forwarding for this connection to have internet on the lab PC).
 2. Make sure you have docker installed. Ideally, install the nvidia docker toolkit as well to run a docker container with Nvidia GPU access if possible (speeds up planning).
 3. Build the docker container `./build_docker_ros.sh user` or if GPU is available `./build_docker_ros.sh user gpu`.
 4. Run the docker container `./run_docker_ros.sh user` or if GPU is available `./run_docker_ros.sh user gpu`.
 5. ```
    catkin build
    source devel/setup.bash
    cd src/stap-ros-pkg
    ```

## Running the Experiments
### Phase 1: Demonstrating the safety shield with different settings
 - The participant wears the marker suite. The participant has to stay behind the line and do not reach above it!
 - Make sure the robot is in blue mode.
 - Partner B: On the *lab PC*:
    1. Open the `CommandsForRealWorldExperiments.md` file.
    2. Open two consoles.
    3. In the first console copy-paste command `1.1.1 Beginner parameters` and execute.
    4. In the second console copy-paste command `1.2 Launching the example motion` and execute.
    5. Demonstrate the shield: Currently the robot slows down if the human is close.
    6. Repeat 3.-5. with the `1.1.2. Intermediate parameters` and `1.1.3. Expert parameters`.
    7. Let the users fill out the first part of the [form](https://docs.google.com/forms/d/e/1FAIpQLSfkcrM-pZIDTupZSPF4uZ3kpkSjptmCOHNRKgOpXUtLXgL_pw/viewform?usp=sf_link).

### Phase 2: Screwdriver Handover task *without* motion preferences
The robot picks the screwdriver, where it's the most feasible and hands the object over in a position, where there is a high chance that the inverse kinematics is successful.
 - Partner A wears the marker suite and positions himself at the desk.
 - Make sure the robot is in blue mode.
 - Place the screwdriver on the little black foam block with the handle pointing the door, the two marker pads should be up. This guarantees good detection and high planning feasibility. Other poses are possible though.
 - Partner B: 
    1. On the *lab PC*: Copy-paste the command the user preferred the most in the previous phase, e.g., `1.1.2. Intermediate parameters`, and execute.
    2. On the *STAP Laptop* in the docker container run:
        ```
        python scripts/eval/eval_planners.py  --planner-config configs/pybullet/planners/policy_cem_screwdriver_no_custom_fns.yaml --env-config configs/pybullet/envs/official/real_domains/screwdriver_handover/task0.yaml --policy-checkpoints models/policies_irl/pick/final_model.pt models/policies_irl/place/final_model.pt models/policies_irl/static_handover/final_model.pt --dynamics-checkpoint models/dynamics_irl/pick_place_static_handover_dynamics/final_model.pt --seed 0 --gui 1 --closed-loop 1 --num-eval 1 --path plots/planning/screwdriver_handover/task0 --verbose 1
        ```

### Phase 3: Screwdriver Handover task *with* motion preferences
The robot picks the screwdriver at the rod and hands the object over in a position, so that the handle points towards the human.
Same as Phase 2 but execute this on *STAP Laptop* instead:
```
python scripts/eval/eval_planners.py  --planner-config configs/pybullet/planners/policy_cem_screwdriver_custom_fns.yaml --env-config configs/pybullet/envs/official/real_domains/screwdriver_handover/task0.yaml --policy-checkpoints models/policies_irl/pick/final_model.pt models/policies_irl/place/final_model.pt models/policies_irl/static_handover/final_model.pt --dynamics-checkpoint models/dynamics_irl/pick_place_static_handover_dynamics/final_model.pt --seed 0 --gui 1 --closed-loop 1 --num-eval 1 --path plots/planning/screwdriver_handover/task0 --verbose 1
```