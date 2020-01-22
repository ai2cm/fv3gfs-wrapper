IMAGE=test-image
KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json
LOCAL_DIR_ARGS = -w /code -v $(shell pwd):/code

RUN_ARGS = $(KEY_ARGS) $(LOCAL_DIR_ARGS) $(IMAGE)


build:
	docker build . -t $(IMAGE)

dev:
	docker run -ti $(RUN_ARGS) bash

test_run_sklearn:
	docker run $(RUN_ARGS) python3 run_sklearn.py
