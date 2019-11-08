#!/bin/bash

if [ -z "$FV3GFS_BUILD_DIR" ]; then
    echo "FV3GFS_BUILD_DIR must be set"
    exit 1
fi

IMAGE=fv3gfs-python

MOUNTS="-v ${FV3GFS_BUILD_DIR}:/FV3 \
        -v ${FV3GFS_BUILD_DIR}/conf/configure.fv3.gnu_docker:/FV3/conf/configure.fv3 \
        -v $(pwd):/cython_wrapper"

docker run --rm $MOUNTS -w /cython_wrapper -it $IMAGE bash
