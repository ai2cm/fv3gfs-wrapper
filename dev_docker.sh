#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3gfs-python

MOUNTS="-v $(pwd):/fv3gfs-python"

CONF_DIR=./lib/external/FV3/conf/

cp $CONF_DIR/configure.fv3.gnu_docker $CONF_DIR/configure.fv3
docker run --rm $MOUNTS -w /fv3gfs-python -it $IMAGE bash
