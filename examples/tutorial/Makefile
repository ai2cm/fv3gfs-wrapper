WRAPPER_IMAGE ?= gcr.io/vcm-ml/fv3gfs-wrapper:v0.5.0-notebook_example
NOTEBOOK_DIR ?= $(shell pwd)/notebooks

JUPYTER_IMAGE = fv3gfs_wrapper_tutorial
DOCKERFILE = Dockerfile
BUILD_ARGS = --build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE)
TEST_TIMEOUT_SECONDS = 240  # took 2:20 for me but depends on connection speed, downloads run directory

ZENODO_RUNDIR_URL = https://zenodo.org/record/4429298/files/c48_6h.tar.gz?download=1
RUNDIR_ARCHIVE = rundir.tar.gz

all: notebooks/reference_rundir build run

run:
	docker run -p 8870:8870 -v $(NOTEBOOK_DIR):/notebooks $(JUPYTER_IMAGE)

build:
	docker build -f $(DOCKERFILE) -t $(JUPYTER_IMAGE) $(BUILD_ARGS) . --target notebook

notebooks/reference_rundir:
	wget $(ZENODO_RUNDIR_URL) -O $(RUNDIR_ARCHIVE)
	tar -xvzf $(RUNDIR_ARCHIVE)
	mv rundir notebooks/reference_rundir
	rm $(RUNDIR_ARCHIVE)
