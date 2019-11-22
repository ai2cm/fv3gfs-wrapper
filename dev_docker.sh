#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3gfs-python

MOUNTS="-v $(pwd):/fv3gfs-python"

docker run --rm $MOUNTS -w /fv3gfs-python -it $IMAGE bash
