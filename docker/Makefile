GCR_URL ?= us.gcr.io/vcm-ml
TARGET ?= fv3gfs-wrapper
TAG_NAME ?= latest
DOCKERFILE ?= Dockerfile
FORTRAN_ROOT ?= ../lib/external
FORTRAN_DOCKERFILE ?= $(FORTRAN_ROOT)/docker/Dockerfile
BUILD_FROM_INTERMEDIATE ?= n

FV3GFS_WRAPPER_IMAGE = $(GCR_URL)/$(TARGET):$(TAG_NAME)
FMS_IMAGE ?= $(GCR_URL)/fms-build
ESMF_IMAGE ?= $(GCR_URL)/esmf-build

FORTRAN_IMAGE = fv3gfs-fortran-build

MOUNTS?=-v $(shell pwd)/FV3:/FV3 \
	-v $(shell pwd)/FV3/conf/configure.fv3.gnu_docker:/FV3/conf/configure.fv3

MOUNTS_SERIALIZE?=-v $(shell pwd)/FV3:/FV3/original

EXPERIMENT ?= new
RUNDIR_CONTAINER=/rundir
RUNDIR_HOST=$(shell pwd)/rundir

FV3GFS_WRAPPER_BUILD_ARGS = --build-arg FMS_IMAGE=$(FMS_IMAGE) --build-arg ESMF_IMAGE=$(ESMF_IMAGE) --build-arg FORTRAN_ENV_IMAGE=$(FORTRAN_IMAGE)

all: build_deps build

build:
	COMPILED_IMAGE=$(FORTRAN_IMAGE) COMPILE_TARGET=fv3gfs-build $(MAKE) -C ../lib/external build
	docker build -f $(DOCKERFILE) -t $(FV3GFS_WRAPPER_IMAGE) $(FV3GFS_WRAPPER_BUILD_ARGS) .. --target $(TARGET)

build_deps:
ifeq ($(BUILD_FROM_INTERMEDIATE),y)
		$(MAKE) pull_deps
else
		docker build -f $(FORTRAN_DOCKERFILE) --target fv3gfs-fms -t $(FMS_IMAGE) $(FORTRAN_ROOT)
		docker build -f $(FORTRAN_DOCKERFILE) --target fv3gfs-esmf -t $(ESMF_IMAGE) $(FORTRAN_ROOT)
endif

push_deps:
	docker push $(FMS_IMAGE)
	docker push $(ESMF_IMAGE)

pull_deps:
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)

.PHONY: build build_deps push_deps pull_deps
