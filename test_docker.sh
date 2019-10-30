#!/bin/bash

DOCKER_IMAGE=fv3gfs-python

./build_docker.sh && docker run -it $DOCKER_IMAGE bash -c "ulimit -s unlimited; cd /cython_wrapper/tests/; ./run_tests.sh"
