#!/bin/bash

DOCKER_IMAGE=us.gcr.io/vcm-ml/fv3gfs-python

./build_docker.sh && docker run -it $DOCKER_IMAGE bash -c "cd /fv3gfs-python; make test"
