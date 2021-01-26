#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3gfs-wrapper:gnu7-mpich314-nocuda

MOUNTS="-v $(pwd):/fv3gfs-wrapper"

CONF_DIR=./lib/external/FV3/conf/

cp $CONF_DIR/configure.fv3.gnu_docker $CONF_DIR/configure.fv3
docker run --rm $MOUNTS -w /fv3gfs-wrapper -it $IMAGE bash
