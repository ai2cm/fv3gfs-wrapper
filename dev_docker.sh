#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3gfs-python:latest

MOUNTS="-v $(pwd)/fv3gfs:/fv3gfs-python/fv3gfs -v $(pwd)/external:/fv3gfs-python/external -v $(pwd)/tests:/fv3gfs-python/tests -v $(pwd)/templates:/fv3gfs-python/templates"

docker run --rm $MOUNTS -w /fv3gfs-python -it $IMAGE bash
