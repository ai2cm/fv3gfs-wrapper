#!/bin/bash

DIGEST=sha256:4870c4e91bc5988df222631f27967572bff977be9ddce6195a1260f7192d85a0
IMAGE=us.gcr.io/vcm-ml/fv3gfs-python@$DIGEST

MOUNTS="-v $(pwd):/fv3gfs-python"

key=/key.json

function run {

docker run --rm -v $(pwd):/code -w /code -e GOOGLE_APPLICATION_CREDENTIALS=$key \
        -v $GOOGLE_APPLICATION_CREDENTIALS:$key \
        -it $IMAGE $@
}
    
run python3 download_rundir.py
run mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 save_state_runfile.py
