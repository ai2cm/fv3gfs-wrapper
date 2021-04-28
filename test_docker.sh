#!/bin/bash

set -e
set -x

if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-wrapper
fi

pytest ./tests/image_tests/*.py

if [[ "GOOGLE_APPLICATION_CREDENTIALS" == "" ]]
then
    docker run -it $DOCKER_IMAGE make -C /fv3gfs-wrapper test
else
    docker run -v $GOOGLE_APPLICATION_CREDENTIALS:$GOOGLE_APPLICATION_CREDENTIALS \
        --env GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
        -it $DOCKER_IMAGE make -C fv3gfs-wrapper test
fi
