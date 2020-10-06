#!/bin/bash

set -ex

if [ -z "$DOCKER_IMAGE" ] ; then
    DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-wrapper:gnu9-mpich314-nocuda
fi

pytest ./tests/image_tests/*.py

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] ; then
    docker run -it "$DOCKER_IMAGE" bash -c "cd /fv3gfs-wrapper; make test"
else
    docker run -v $GOOGLE_APPLICATION_CREDENTIALS:$GOOGLE_APPLICATION_CREDENTIALS \
        --env GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
        -it $DOCKER_IMAGE bash -c "cd /fv3gfs-wrapper; make test"
fi
