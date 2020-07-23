#!/bin/bash

set -e

##
## Select GNU and MPI versions (uncomment only 1 line)
##-----------------------------------------------------

export HPC_CONFIG=gnu9_mpich314_nocuda
#export HPC_CONFIG=gnu8_mpich314_nocuda
#export HPC_CONFIG=gnu8_mpich314_cuda101

##
## Give the final name & destination of the built fv3gfs-python container
##-----------------------------------------------------------------------

OUTPUT_IMAGE=us.gcr.io/vcm-ml/fv3gfs-python:${HPC_CONFIG}

#
#===============  Do not change anything below this line ================
#

# Prepare the fv3gfs-python source tarball
rm -f fv3gfs-python.tar
tar cvf fv3gfs-python.tar HISTORY.md Makefile docker external setup.cfg tests LICENSE README.md docs fill_templates.py lib setup.py_hpc_gnu templates tox.ini MANIFEST.in RELEASE.rst dev_docker.sh examples fv3gfs requirements.txt 

# Build the requested Docker image
export DOCKER_BUILDKIT=1
export CONTAINER=docker/Dockerfile.${HPC_CONFIG}

docker build -f ${CONTAINER} --target fv3-python-bld -t ${OUTPUT_IMAGE} .

