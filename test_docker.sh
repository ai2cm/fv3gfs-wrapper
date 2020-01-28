#!/bin/bash

set -e

DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-python

./build_docker.sh

if [[ "GOOGLE_APPLICATION_CREDENTIALS" == "" ]]
    docker run -it $DOCKER_IMAGE bash -c "pytest /fv3gfs-python/external/fv3config/tests"
else
# needed for circleci tests on machine executor, even though we're accessing public data only
    docker run -v $GOOGLE_APPLICATION_CREDENTIALS:$GOOGLE_APPLICATION_CREDENTIALS \
        --env GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
        -it $DOCKER_IMAGE bash -c "pytest /fv3gfs-python/external/fv3config/tests"
fi

docker run -it $DOCKER_IMAGE bash -c "cd /fv3gfs-python; make test"
