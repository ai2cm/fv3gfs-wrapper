#!/bin/bash

set -e

if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-python
fi

# if [ ! -z "$DOCKER_BUILD_ARGS" ]; then
#     for arg in $(< $DOCKER_BUILD_ARGS); do
#         build_arg="$build_arg --build-arg $arg"
#     done
# fi


docker build -f lib/external/docker/Dockerfile --target fv3gfs-environment -t fv3gfs-fortran-env lib/external && \
    docker build -f lib/external/docker/Dockerfile --target fv3gfs-fms -t fv3gfs-fms lib/external && \
    docker build -f lib/external/docker/Dockerfile --target fv3gfs-esmf -t fv3gfs-esmf lib/external && \
    docker build -f lib/external/docker/Dockerfile --target fv3gfs-build -t fv3gfs-fortran-build lib/external && \
    docker build -f docker/Dockerfile -t $DOCKER_IMAGE $build_arg . --target fv3gfs-python
