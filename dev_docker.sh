#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3gfs-wrapper:latest

MOUNTS="-v $(pwd)/fv3gfs:/fv3gfs-wrapper/fv3gfs \
        -v $(pwd)/external:/fv3gfs-wrapper/external -v $(pwd)/lib:/fv3gfs-wrapper/lib -v $(pwd)/tests:/fv3gfs-wrapper/tests -v $(pwd)/templates:/fv3gfs-wrapper/templates\
	-v $(pwd)/examples:/fv3gfs-wrapper/examples"

CONF_DIR=./lib/external/FV3/conf/

cp $CONF_DIR/configure.fv3.gnu_docker $CONF_DIR/configure.fv3
docker run --rm $MOUNTS -w /fv3gfs-wrapper -it $IMAGE bash
