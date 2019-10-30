#!/bin/bash

DOCKER_IMAGE=fv3gfs-python

docker build -f docker/Dockerfile -t $DOCKER_IMAGE .
