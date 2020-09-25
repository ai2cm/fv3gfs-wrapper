GCR_URL ?= us.gcr.io/vcm-ml
TARGET ?= fv3gfs-wrapper
TAG_NAME ?= latest
DOCKERFILE ?= Dockerfile
FORTRAN_ROOT ?= ../lib/external
FORTRAN_DOCKERFILE ?= $(FORTRAN_ROOT)/docker/Dockerfile
BUILD_FROM_INTERMEDIATE ?= y

DEP_TAG_NAME ?= gnu9-mpich314-nocuda
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

FV3GFS_WRAPPER_BUILD_ARGS = --build-arg FMS_IMAGE=$(FMS_IMAGE) --build-arg ESMF_IMAGE=$(ESMF_IMAGE) --build-arg FORTRAN_ENV_IMAGE=$(FORTRAN_IMAGE) --build-arg MPI_IMAGE=$(MPI_IMAGE)

all: build_deps build

build:
	DOCKER_BUILDKIT=1 COMPILED_IMAGE=$(FORTRAN_IMAGE) COMPILE_TARGET=fv3gfs-build BUILD_FROM_INTERMEDIATE=$(BUILD_FROM_INTERMEDIATE) $(MAKE) -C ../lib/external build
	docker build -f $(DOCKERFILE) -t $(FV3GFS_WRAPPER_IMAGE) $(FV3GFS_WRAPPER_BUILD_ARGS) .. --target $(TARGET)

build_deps:
ifeq ($(BUILD_FROM_INTERMEDIATE),y)
		$(MAKE) pull_deps
else
	docker build -f $(FORTRAN_DOCKERFILE).full --target fv3gfs-mpi -t $(MPI_IMAGE) $(FORTRAN_ROOT)
	docker build -f $(FORTRAN_DOCKERFILE) --target fv3gfs-fms -t $(FMS_IMAGE) $(FORTRAN_ROOT)
	docker build -f $(FORTRAN_DOCKERFILE) --target fv3gfs-esmf -t $(ESMF_IMAGE) $(FORTRAN_ROOT)
endif

push_deps:
	docker push $(MPI_IMAGE)
	docker push $(FMS_IMAGE)
	docker push $(ESMF_IMAGE)

pull_deps:
	docker pull $(MPI_IMAGE)
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)

.PHONY: build build_deps push_deps pull_deps
