#!/bin/bash

if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE = fv3gfs-python
fi

for arg in $(< $DOCKER_BUILD_ARGS); do
    build_arg = $build_arg --build-arg $arg
done

docker build -f docker/Dockerfile -t $DOCKER_IMAGE $build_arg .
