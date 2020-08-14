#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3gfs-python:latest

MOUNTS="-v $(pwd)/fv3gfs:/fv3gfs-python/fv3gfs \
        -v $(pwd)/external:/fv3gfs-python/external -v $(pwd)/lib:/fv3gfs-python/lib -v $(pwd)/tests:/fv3gfs-python/tests -v $(pwd)/templates:/fv3gfs-python/templates\
	-v $(pwd)/examples:/fv3gfs-python/examples"
MOUNTS="-v $(pwd):/fv3gfs-python"

CONF_DIR=./lib/external/FV3/conf/

cp $CONF_DIR/configure.fv3.gnu_docker $CONF_DIR/configure.fv3
docker run --rm $MOUNTS -w /fv3gfs-python -it $IMAGE bash
