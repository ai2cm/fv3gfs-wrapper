.. highlight:: shell

============
Installation
============


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


Stable release
--------------

To install `fv3gfs`, run this command in your terminal:

.. code-block:: console

    $ pip install fv3gfs

This is the preferred method to install `fv3gfs`, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for `fv3gfs` can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/VulcanClimateModeling/fv3gfs

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/VulcanClimateModeling/fv3gfs/tarball/master

Once you have a copy of the source, you can install it with the following steps:

1. Build the [Fortran code](https://github.com/VulcanClimateModeling/fv3gfs/tree/master/sorc/fv3gfs.fd/FV3)
   for the FV3GFS model. Build artifacts (files ending in `.o` and `.a`) should remain in the directory.

2. `export FV3GFS_BUILD_DIR = <FV3 directory>`

3. From within the Python wrapper root directory, run `make dist` and `pip install .`.

4. If the build worked correctly, the package should now be installed. If there were issues in compiling the
   wrapper, there may be an incompatability between `setup.py` and your system.
   [Open an issue](https://github.com/VulcanClimateModeling/fv3gfs/issues/new) on Github.


.. _Github repo: https://github.com/VulcanClimateModeling/fv3gfs
.. _tarball: https://github.com/VulcanClimateModeling/fv3gfs/tarball/master
