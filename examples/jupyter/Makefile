WRAPPER_IMAGE ?= us.gcr.io/vcm-ml/fv3gfs-wrapper:gnu9-mpich314-nocuda
NOTEBOOK_DIR ?= $(shell pwd)/notebooks

JUPYTER_IMAGE = fv3gfs_jupyter
DOCKERFILE = Dockerfile
BUILD_ARGS = --build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE)
TEST_TIMEOUT_SECONDS = 240  # took 2:20 for me but depends on connection speed, downloads run directory

all: build run

run:
	docker run -p 8888:8888 -v $(NOTEBOOK_DIR):/notebooks $(JUPYTER_IMAGE)

build:
	docker build -f $(DOCKERFILE) -t $(JUPYTER_IMAGE) $(BUILD_ARGS) . --target notebook

test:
	timeout $(TEST_TIMEOUT_SECONDS) docker run -v $(NOTEBOOK_DIR):/notebooks $(JUPYTER_IMAGE) jupyter nbconvert --to notebook --execute /notebooks/random_forest.ipynb
