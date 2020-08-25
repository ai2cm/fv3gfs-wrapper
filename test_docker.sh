#!/bin/bash

set -e

if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-wrapper
fi

pytest ./tests/image_tests/*.py
