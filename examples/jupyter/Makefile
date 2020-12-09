WRAPPER_IMAGE ?= us.gcr.io/vcm-ml/fv3gfs-wrapper:gnu9-mpich314-nocuda
NOTEBOOK_DIR ?= $(shell pwd)/notebooks

JUPYTER_IMAGE = fv3gfs_jupyter
DOCKERFILE = Dockerfile
BUILD_ARGS = --build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE)

all: build run

run:
	docker run -p 8888:8888 -v $(NOTEBOOK_DIR):/notebooks $(JUPYTER_IMAGE)

build:
	docker build -f $(DOCKERFILE) -t $(JUPYTER_IMAGE) $(BUILD_ARGS) . --target notebook
