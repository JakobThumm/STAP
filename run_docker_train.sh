# Run this file either with `./run_docker_train_v2.sh user` or `./run_docker_train_v2.sh root`.
# User mode ensures that the files you create are not made by root.
# Root mode creates a "classic" root user in docker.
# The /runs, /models, and /wandb folders are mounted 
# to store training results outside the docker.

user=${1:-user}
gpu=${2:-cpu}
bash_command="/bin/bash"
docker_command="docker run -it"

if [ -n "$3" ]
then
    bash_command="${3}"
    docker_command="docker run -d"
fi

command="${bash_command}"

echo "Chosen mode: $user, chosen gpu: $gpu, chosen command: $command"
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

echo "Running docker command: ${docker_command} ${options} ${image} ${command}"

${docker_command} \
    ${options} \
    ${image} \
    ${command}