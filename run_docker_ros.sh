# Run this file either with `./run_docker_train_v2.sh user` or `./run_docker_train_v2.sh root`.
# User mode ensures that the files you create are not made by root.
# Root mode creates a "classic" root user in docker.
# The /runs, /models, and /wandb folders are mounted 
# to store training results outside the docker.

user=${1:-user}
gpu=${2:-gpu}
bash_command="/bin/bash"
docker_command="docker run -it"

if [ -n "$3" ]
then
    bash_command="${3}"
    docker_command="docker run -d"
fi

command="${bash_command}"

echo "Chosen mode: $user, chosen command: $command"
options="--shm-size=10.24gb 
         --privileged 
         --init 
         --net=host 
         -v /tmp/.X11-unix:/tmp/.X11-unix 
         -e DISPLAY=$DISPLAY 
         -h $HOSTNAME 
         -v $HOME/.Xauthority:/root/.Xauthority"
         -e ROS_MASTER_URI=http://10.42.0.69:11311
         -e ROS_IP=10.42.0.1
         -e ROS_HOSTNAME=10.42.0.1

image="stap-ros"

if [ "$gpu" = "gpu" ]
then
    options="${options} --gpus all"
    image="${image}-gpu"
fi

if [ "$user" = "root" ]
    then
    options="${options} 
        --volume="$(pwd)/models/:/root/catkin_ws/src/stap-ros-pkg/models/" 
        --volume="$(pwd)/CMakeLists.txt:/root/catkin_ws/src/stap-ros-pkg/CMakeLists.txt"
        --volume="$(pwd)/package.xml:/root/catkin_ws/src/stap-ros-pkg/package.xml" 
        --volume="$(pwd)/setup.py:/root/catkin_ws/src/stap-ros-pkg/setup.py" 
        --volume="$(pwd)/stap/:/root/catkin_ws/src/stap-ros-pkg/stap/" 
        --volume="$(pwd)/configs/:/root/catkin_ws/src/stap-ros-pkg/configs/" 
        --volume="$(pwd)/scripts/:/root/catkin_ws/src/stap-ros-pkg/scripts/" 
        --volume="$(pwd)/launch/:/root/catkin_ws/src/stap-ros-pkg/launch/"
        --volume="$(pwd)/prompts/:/root/catkin_ws/src/stap-ros-pkg/prompts/"
    "
    image="${image}/root:v2"
elif [ "$user" = "user" ]
    then
    options="${options} 
        --volume="$(pwd)/models/:/home/$USER/catkin_ws/src/stap-ros-pkg/models/" 
        --volume="$(pwd)/CMakeLists.txt:/home/$USER/catkin_ws/src/stap-ros-pkg/CMakeLists.txt"
        --volume="$(pwd)/package.xml:/home/$USER/catkin_ws/src/stap-ros-pkg/package.xml" 
        --volume="$(pwd)/setup.py:/home/$USER/catkin_ws/src/stap-ros-pkg/setup.py" 
        --volume="$(pwd)/stap/:/home/$USER/catkin_ws/src/stap-ros-pkg/src/stap/" 
        --volume="$(pwd)/configs/:/home/$USER/catkin_ws/src/stap-ros-pkg/configs/" 
        --volume="$(pwd)/scripts/:/home/$USER/catkin_ws/src/stap-ros-pkg/scripts/" 
        --volume="$(pwd)/launch/:/home/$USER/catkin_ws/src/stap-ros-pkg/launch/" 
        --volume="$(pwd)/prompts/:/home/$USER/catkin_ws/src/stap-ros-pkg/prompts/" 
        --user=$USER"
    image="${image}/$USER:v2"
else
    echo "User mode unknown. Please choose user, root, or leave out for default user"
fi

echo "Running docker command: ${docker_command} ${options} ${image} ${command}"

${docker_command} \
    ${options} \
    ${image} \
    ${command}