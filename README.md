
FV3GFS
======

fv3gfs-python (`import fv3gfs`), is a Python wrapper for the FV3GFS
global climate model.

Checking out
------------

This package uses submodules. After you check out the repository, you must run
`git submodule init` followed by `git submodule update` in the root directory of this package.


* Free software: BSD license

Dependencies
------------

`fv3gfs` depends on a number of development libraries that can't be installed through `pip`.
On Ubuntu, these can be installed with:

.. code-block:: console

    apt-get update && apt-get install -y \
        libblas-dev \
        liblapack-dev \
        libnetcdf-dev \
        libnetcdff-dev \
        libffi-dev \
        libopenmpi3 \
        libssl-dev 

The Fortran installation located under `lib/FV3` is used by default. This can be
overridden by setting `FV3GFS_BUILD_DIR`. It is assumed that the build system under
`FV3GFS_BUILD_DIR` is the same.

Installation
------------

The Docker image can be built using `build_docker.sh`, or built and then
tested using `test_docker.sh` (which will use the existing build if present).

On a host, the package can be built using `make build`, and then installed
in development mode with `pip install -e .`.

This package only supports linux and Python 3.5 or greater.

Quickstart
----------

Once the docker image is built, you could enter it and run the online code example using:

.. code-block:: python

    docker run -it fv3gfs-python bash
    cd /fv3gfs-python/examples
    ulimit -s unlimited
    mpirun -np 6 --oversubscribe --allow-run-as-root --mca btl_vader_single_copy_mechanism none python online_code.py

Usage
-----

Example run scripts are included in [`examples`](https://github.com/VulcanClimateModeling/fv3gfs/tree/master/sorc/fv3gfs.fd/cython_wrapper/examples).
These run scripts act as a drop-in replacement for `fv3.exe`, and get executed
in the same way, using `mpirun`.
