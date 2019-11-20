#!/bin/bash

if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-python:latest
fi

if [ ! -z "$DOCKER_BUILD_ARGS"]; then
    for arg in $(< $DOCKER_BUILD_ARGS); do
        build_arg="$build_arg --build-arg $arg"
    done
fi

docker build -f docker/Dockerfile -t $DOCKER_IMAGE $build_arg .
