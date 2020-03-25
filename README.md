
FV3GFS
======

fv3gfs-python (`import fv3gfs`), is a Python wrapper for the FV3GFS
global climate model.

See the documentation for more detailed instructions.

Checking out
------------

This package uses submodules. After you check out the repository, you must run
`git submodule update --init --recursive` in the root directory of this package.

* Free software: BSD license

Installation
------------

The Docker image can be built using `build_docker.sh`, or built and then
tested using `test_docker.sh` (which will use the existing build if present).
The first time you build, both ESMF and FMS will be
built, taking up quite a lot of time. On subsequent builds, these may be retrieved
from cached images, if you allow caching on your system.

On a host, the package can be built using `make build`, and then installed
in development mode with `pip install -e .`.

This package only supports linux and Python 3.5 or greater.

Iterative development
---------------------

When making changes to the Fortran source code, you may want to rebuild just part of
the model, keeping build artifacts between rebuilds. You can do this by running the
docker image with bind-mounts into your local filesystem. Just be sure to `make clean`
when you're done to remove the build artifacts, or it may cause problems when you
build the docker image.

With the image already built by `build_docker.sh` or pulled using
`docker pull us.gcr.io/vcm-ml/fv3gfs-python`, run `dev_docker.sh`. This will
bind-mount the `fv3gfs`, `lib`, `tests`, `external`, and `templates` directories into the
docker image. Inside the docker image, you can build or re-build the model with
`make build` inside the `/fv3gfs-python` directory, and run the test suite with
`make test`.

Building Docs
-------------

Once the docker image is built, the documentation can be built and shown using:

    make docs-docker

This will produce html documentation in `docs/html`.

Usage
-----

Example run scripts are included in [`examples/runfiles`](https://github.com/VulcanClimateModeling/fv3gfs/tree/master/sorc/fv3gfs.fd/cython_wrapper/examples/runfiles).
These run scripts act as a drop-in replacement for `fv3.exe`, and get executed
in the same way, using `mpirun`:

    mpirun -np 6 --oversubscribe --allow-run-as-root --mca btl_vader_single_copy_mechanism none python online_code.py

Running these files requires them to be placed inside a valid run directory. This is
done automatically if you run them using `fv3run`, as is done in
the Makefile in that directory.
