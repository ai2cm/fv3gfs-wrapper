.. highlight:: shell

Developers Notes
================

Here are some fairly informal notes about the build system and package organization
structure for the ``fv3gfs-wrapper`` wrapper.

``pace-util`` Package
-----------------------

Much of the functionality used to write run scripts in ``fv3gfs-wrapper`` is split out
into a package called ``pace-util``.
Any functionality that can be written to be independent of the wrapped Fortran model
is put into ``pace-util``. This makes it much easier to iterate on and test pure Python
code for the model, as one can install the package onto the host system without needing
to use Docker to handle Fortran dependencies. This also makes it possible for analysis
codes to use the same routines used by the wrapped model.

We recommend users and developers read the `pace-util` documentation as well to make
full use of this package.

Templates
---------

Several Fortran Jinja "template" files exist in ``templates``. During model build, these are
converted into Fortran sources which are then compiled.

Properties Files
----------------

The build system for fv3gfs-wrapper uses "properties" files which list Fortran model 
variables. These do not handle "tracer" properties, which need to be determined at 
run-time by querying the model, since these are determined by the field table.

While building the model, these properties are used by the Jinja templates to
automatically write getters and setters for each model variable. This allows
:py:func:`fv3gfs.wrapper.get_state` and :py:func:`fv3gfs.wrapper.set_state` to interface with those
variables in the Fortran model. Getting/setting new variables involves updating those
properties (in their respective JSON files) and then rebuilding the wrapper.

Docker build system
-------------------

`fv3gfs-wrapper` depends on a number of docker build stages, and is itself used to
build other images. To avoid these docs being out of date, you can see exactly which
stages are depended on by reading `build_docker.sh`.

Images built using a dockerfile under `lib/external` are provided by the fv3gfs-fortran
repo. Compiled binaries from these images are copied into the fv3gfs-wrapper image.

Building using Docker is a little all-or-nothing. For example, to rebuild the image
after making a small change to the Fortran code requires entirely re-building the
Fortran model. This can be avoided to some degree by using `dev_docker.sh` to enter
the image interactively.

Intermediate docker images
--------------------------

For certain dependencies, images are pre-generated and pulled from GCR by default.
This prevents having to re-build dependencies which rarely change. As of writing there
is no versioning on these dependencies.

To perform a default fv3gfs-wrapper build which pulls images from GCR, in the root directory run::

    $ make build-docker

To rebuild the intermediate images locally, run::

    $ BUILD_FROM_INTERMEDIATE=n make -C docker build_deps

To push built images to GCR::

    $ make -C docker push_deps

To perform a fv3gfs-wrapper build while building all steps locally, run::

    $ BUILD_FROM_INTERMEDIATE=n make -C docker build

Building all steps locally takes longer, but will ensure all built steps are consistent.
This is required in order to create builds which depend on older versions of the
dependencies included in intermediate images.
