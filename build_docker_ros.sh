# Run this file either with `./build_docker_train_v2.sh user` or `./build_docker_train_v2.sh root`.
# User mode will create a user in the docker so that the files you create are not made by root.
# Root mode creates a "classic" root user in docker.

user=${1:-user}
gpu=${2:-gpu}
cache=${3:-cache}
echo "Chosen mode: $user"

command="docker build"
options=""
dockerfile="Dockerfile"
image="stap-ros"

if [ "$gpu" = "gpu" ]
then
    image="${image}-gpu"
    # dockerfile="${dockerfile}121.gpu"
    dockerfile="${dockerfile}118.ros"
else
    dockerfile="${dockerfile}.ros"
fi

if [ "$user" = "root" ]
    then
    options="${options} --build-arg MODE=root"
    image="${image}/root:v2"
elif [ "$user" = "user" ]
    then
    options="${options} --build-arg MODE=user \
        --build-arg USER_UID=$(id -u) \
        --build-arg USER_GID=$(id -g) \
        --build-arg USERNAME=$USER"
    image="${image}/$USER:v2"
else
    echo "User mode unknown. Please choose user, root, or leave out for default user"
fi

if [ "$cache" = "nocache" ]
    then
    options="${options} --no-cache"
fi

echo "Running docker command: ${command} ${options} -f ${dockerfile} -t ${image} ."
DOCKER_BUILDKIT=1
${command} ${options} -f ${dockerfile} -t ${image} .
