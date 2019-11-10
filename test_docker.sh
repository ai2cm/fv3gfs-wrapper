#!/bin/bash

DOCKER_IMAGE=fv3gfs-python

./build_docker.sh && docker run -it $DOCKER_IMAGE bash -c "cd /fv3gfs-python; make test"
