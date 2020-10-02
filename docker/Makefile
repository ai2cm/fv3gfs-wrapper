GCR_URL ?= us.gcr.io/vcm-ml
TARGET ?= fv3gfs-wrapper
TAG_NAME ?= latest
DOCKERFILE ?= Dockerfile
FORTRAN_ROOT ?= ../lib/external
FORTRAN_DOCKERFILE ?= $(FORTRAN_ROOT)/docker/Dockerfile
BUILD_FROM_INTERMEDIATE ?= n

CUDA ?= n
ifeq ($(CUDA),n)
	BASE_IMAGE ?=ubuntu:19.10
	DEP_TAG_NAME ?= gnu9-mpich314-nocuda
	GT4PY_CUDA = " "
else
	BASE_IMAGE ?=nvidia/cuda:10.2-devel-ubuntu18.04
	DEP_TAG_NAME ?= gnu8-mpich314-cuda102
	GT4PY_CUDA = [cuda]
endif

FV3GFS_WRAPPER_IMAGE = $(GCR_URL)/$(TARGET):$(DEP_TAG_NAME)
MPI_IMAGE = $(GCR_URL)/mpi-build:$(DEP_TAG_NAME)
FMS_IMAGE ?= $(GCR_URL)/fms-build:$(DEP_TAG_NAME)
ESMF_IMAGE ?= $(GCR_URL)/esmf-build:$(DEP_TAG_NAME)

FORTRAN_IMAGE = fv3gfs-fortran-build

MOUNTS?=-v $(shell pwd)/FV3:/FV3 \
	-v $(shell pwd)/FV3/conf/configure.fv3.gnu_docker:/FV3/conf/configure.fv3

MOUNTS_SERIALIZE?=-v $(shell pwd)/FV3:/FV3/original

EXPERIMENT ?= new
RUNDIR_CONTAINER=/rundir
RUNDIR_HOST=$(shell pwd)/rundir

FV3GFS_WRAPPER_BUILD_ARGS  = --build-arg BASE_IMAGE=$(BASE_IMAGE) 
FV3GFS_WRAPPER_BUILD_ARGS += --build-arg FMS_IMAGE=$(FMS_IMAGE) 
FV3GFS_WRAPPER_BUILD_ARGS += --build-arg ESMF_IMAGE=$(ESMF_IMAGE) 
FV3GFS_WRAPPER_BUILD_ARGS += --build-arg FORTRAN_ENV_IMAGE=$(FORTRAN_IMAGE) 
FV3GFS_WRAPPER_BUILD_ARGS += --build-arg MPI_IMAGE=$(MPI_IMAGE)
FV3GFS_WRAPPER_BUILD_ARGS += --build-arg GT4PY_CUDA=$(GT4PY_CUDA)

all: build

build:
	BASE_IMAGE=$(BASE_IMAGE) \
		COMPILED_IMAGE=$(FORTRAN_IMAGE) \
		COMPILE_TARGET=fv3gfs-build \
		BUILD_FROM_INTERMEDIATE=$(BUILD_FROM_INTERMEDIATE) \
		CUDA=$(CUDA) \
		DEP_TAG_NAME=$(DEP_TAG_NAME) \
		GCR_URL=$(GCR_URL) \
		$(MAKE) -C ../lib/external build
	docker build -f $(DOCKERFILE) -t $(FV3GFS_WRAPPER_IMAGE) $(FV3GFS_WRAPPER_BUILD_ARGS) .. --target $(TARGET)

pull_deps:
	docker pull $(MPI_IMAGE)
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)

.PHONY: build pull_deps
