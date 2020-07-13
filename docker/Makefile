GCR_URL ?=
TARGET ?= fv3gfs-python
TAG_NAME ?= latest
FORTRAN_ROOT ?= ../lib/external
BUILD_FROM_INTERMEDIATE ?= n
DOCKERFILE ?= Dockerfile
FORTRAN_DOCKERFILE ?= docker/Dockerfile

FV3GFS_PYTHON_IMAGE = $(GCR_URL)$(TARGET):$(TAG_NAME)
FMS_IMAGE ?= $(GCR_URL)fms-build
ESMF_IMAGE ?= $(GCR_URL)esmf-build
DAWN_IMAGE ?= $(GCR_URL)dawn-build
GT4PY_IMAGE ?= $(GCR_URL)gt4py-build

FMS_TARGET=fv3gfs-fms
ESMF_TARGET=fv3gfs-esmf
FV3_TARGET=fv3gfs-build
SETUP_NAME=setup.py_orig

HPC_BUILD ?= n
ifeq ($(HPC_BUILD),gnu)
	DOCKERFILE = Dockerfile.hpc
	FMS_TARGET = fms-bld
	ESMF_TARGET = esmf-bld
	FV3_TARGET = fv3-bld
	FORTRAN_DOCKERFILE = docker/Dockerfile.gnu8_mpich314_cuda101
        SETUP_NAME=setup.py_hpc_gnu
endif
ifeq ($(HPC_BUILD),intel)
	DOCKERFILE ?= Dockerfile.hpc
	FMS_TARGET = fms-bld
	ESMF_TARGET = esmf-bld
	FV3_TARGET = fv3-bld
	FORTRAN_DOCKERFILE ?= docker/Dockerfile.intel2021_mpich314_nocuda
        SETUP_NAME=setup.py_hpc_intel
endif

FORTRAN_IMAGE = fv3gfs-fortran-build

MOUNTS?=-v $(shell pwd)/FV3:/FV3 \
	-v $(shell pwd)/FV3/conf/configure.fv3.gnu_docker:/FV3/conf/configure.fv3

MOUNTS_SERIALIZE?=-v $(shell pwd)/FV3:/FV3/original

EXPERIMENT ?= new
RUNDIR_CONTAINER=/rundir
RUNDIR_HOST=$(shell pwd)/rundir

FV3GFS_PYTHON_BUILD_ARGS = --build-arg FMS_IMAGE=$(FMS_IMAGE) --build-arg ESMF_IMAGE=$(ESMF_IMAGE) --build-arg DAWN_IMAGE=$(DAWN_IMAGE) --build-arg FORTRAN_ENV_IMAGE=$(FORTRAN_IMAGE)
DAWN_BUILD_ARGS = --build-arg FMS_IMAGE=$(FMS_IMAGE) --build-arg ESMF_IMAGE=$(ESMF_IMAGE) --build-arg FORTRAN_ENV_IMAGE=$(FORTRAN_IMAGE)

all: build_deps build

build:
	DOCKERFILE=$(FORTRAN_DOCKERFILE) COMPILED_IMAGE=$(FORTRAN_IMAGE) COMPILE_TARGET=$(FV3_TARGET) $(MAKE) -C ../lib/external build
	cd .. && cp $(SETUP_NAME) setup.py  && tar cf fv3gfs-python.tar *   
	DOCKER_BUILDKIT=1 docker build -f $(DOCKERFILE) -t $(FV3GFS_PYTHON_IMAGE) $(FV3GFS_PYTHON_BUILD_ARGS) --target fv3gfs-python ..

build_deps:
	DOCKER_BUILDKIT=1 docker build -f $(FORTRAN_ROOT)/$(FORTRAN_DOCKERFILE) --target $(FMS_TARGET) -t $(FMS_IMAGE) $(FORTRAN_ROOT)
	DOCKER_BUILDKIT=1 docker build -f $(FORTRAN_ROOT)/$(FORTRAN_DOCKERFILE) --target $(ESMF_TARGET) -t $(ESMF_IMAGE) $(FORTRAN_ROOT)
	DOCKER_BUILDKIT=1 docker build -f $(DOCKERFILE) --target fv3gfs-dawn -t $(DAWN_IMAGE) .
ifneq ($(HPC_BUILD),n)
	DOCKER_BUILDKIT=1 docker build -f $(DOCKERFILE) --target fv3gfs-gt4py -t $(GT4PY_IMAGE) .
endif

push_deps:
	docker push $(FMS_IMAGE)
	docker push $(ESMF_IMAGE)
	docker push $(DAWN_IMAGE)

pull_deps:
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)
	docker pull $(DAWN_IMAGE)

.PHONY: build build_deps push_deps pull_deps
