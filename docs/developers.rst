.. highlight:: shell

================
Developers Notes
================

Here are some fairly informal notes about the build system and package organization
structure for the ``fv3gfs-python`` wrapper.

fv3util package
---------------

Much of the functionality of ``fv3gfs-python`` is split out into a package called ``fv3util``.
Any functionality that can be written to be independent of the wrapped Fortran model
is put into ``fv3util``. This makes it much easier to iterate on and test pure Python
code for the model, as one can install the package onto the host system without needing
to use Docker to handle Fortran dependencies. This also makes it possible for analysis
codes to use the same routines used by the Fortran model.

Versions of ``fv3util`` are paired with versions of ``fv3gfs-python``. The wrapper
requires a copy of ``fv3util`` with the same version.

templates
---------

Several Fortran Jinja "template" files exist in ``templates``. During model build, these are
converted into Fortran sources which are then compiled.

properties files
----------------

Inside fv3util are "properties" for Fortran model variables. These are separated into
"dynamics", "physics", and "tracer". "tracer" properties don't exist in fv3util itself
but rather have to be registered into the package. This allows the Fortran model to
pass information about the tracers being used with the current run configuration.
Analysis codes can similarly pass tracer properties into fv3util. This is required,
for example, to load restart data that includes tracers.

While building the model, these properties are used by the Jinja templates to
automatically write getters and setters for each model variable. This allows
:py:func:`fv3gfs.get_state` and :py:func:`fv3gfs.set_state` to interface with those
variables in the Fortran model. Getting/setting new variables involves updating those
properties (in their respective JSON files) and then rebuilding the wrapper.
