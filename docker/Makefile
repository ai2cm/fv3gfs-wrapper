GCR_URL ?= us.gcr.io/vcm-ml
TARGET ?= fv3gfs-wrapper
BUILD_FROM_INTERMEDIATE ?= n

DEP_TAG_NAME ?= gnu7-mpich314-nocuda

FV3GFS_WRAPPER_IMAGE = $(GCR_URL)/$(TARGET):$(DEP_TAG_NAME)
MPI_IMAGE = $(GCR_URL)/mpi-build:$(DEP_TAG_NAME)
FMS_IMAGE ?= $(GCR_URL)/fms-build:$(DEP_TAG_NAME)
ESMF_IMAGE ?= $(GCR_URL)/esmf-build:$(DEP_TAG_NAME)
FORTRAN_IMAGE = fv3gfs-fortran-build

all: build

build_fv3gfs_fortran:
	# Currently this build depends on the build artifacts produced by the
	# fv3gfs-fortran submodule
	COMPILED_IMAGE=$(FORTRAN_IMAGE) \
		COMPILE_TARGET=fv3gfs-build \
		BUILD_FROM_INTERMEDIATE=$(BUILD_FROM_INTERMEDIATE) \
		DEP_TAG_NAME=$(DEP_TAG_NAME) \
		GCR_URL=$(GCR_URL) \
		$(MAKE) -C ../lib/external build

build: build_fv3gfs_fortran
	docker build -f Dockerfile \
		-t $(FV3GFS_WRAPPER_IMAGE) \
		--build-arg FMS_IMAGE=$(FMS_IMAGE)  \
		--build-arg ESMF_IMAGE=$(ESMF_IMAGE)  \
		--build-arg FORTRAN_ENV_IMAGE=$(FORTRAN_IMAGE)  \
		--build-arg MPI_IMAGE=$(MPI_IMAGE) \
		--target $(TARGET) \
		..

pull_deps:
	docker pull $(MPI_IMAGE)
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)

.PHONY: build pull_deps
