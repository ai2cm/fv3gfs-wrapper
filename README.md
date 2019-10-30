
FV3GFS
======


`fv3gfs` is a Python wrapper for the FV3GFS global climate model.


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

When installing from sources, it also requires that you build the FV3GFS Fortran model prior
to building the Python package.


Installation
------------

1. Build the [Fortran code](https://github.com/VulcanClimateModeling/fv3gfs/tree/master/sorc/fv3gfs.fd/FV3)
   for the FV3GFS model. Build artifacts (files ending in `.o` and `.a`) should remain in the directory.
2. `export FV3GFS_BUILD_DIR = <FV3 directory>`
3. From within the Python wrapper root directory, run `make dist` and `pip install .`.
4. If the build worked correctly, the package should now be installed. If there were issues
   in compiling the wrapper, there may be an incompatability between `setup.py` and your system.
   [Open an issue](https://github.com/VulcanClimateModeling/fv3gfs/issues/new) on Github.

Currently, this package only supports linux and Python 3.5 or greater.


Docker Installation
-------------------

The docker image can be built using `build_docker.sh`, or directly using the Dockerfile
in the `docker` folder. It is based off of the `fv3gfs-compiled-default` image
([github](https://github.com/VulcanClimateModeling/fv3gfs)), which must be built first.


Quickstart
----------

Once the docker image is built, you could enter it and run the online code example using:

.. code-block:: python

    docker run -it fv3gfs-python bash
    cd /cython_wrapper/examples
    ulimit -s unlimited
    mpirun -np 6 --oversubscribe --allow-run-as-root --mca btl_vader_single_copy_mechanism none python online_code.py

Usage
-----

Example run scripts are included in [`examples`](https://github.com/VulcanClimateModeling/fv3gfs/tree/master/sorc/fv3gfs.fd/cython_wrapper/examples).
These run scripts act as a drop-in replacement for `fv3.exe`, and get executed
in the same way, using `mpirun`.

Features
--------

* TODO
