=====
Usage
=====

Using MPI
---------

This model uses MPI, and must be executed using :code:`mpirun`. The basic command would be something like::

    $ mpirun -n 6 model.py

Adding the flag :code:`--mca btl_vader_single_copy_mechanism none` will silence some (seemingly inconsequential)
errors that occur when running like this (these appear also in the pure Fortran model). If you're running
as root inside a Docker container, you need to add :code:`--allow-run-as-root`
(this should be avoided outside of Docker).

You also need to ensure that when the script executes model subroutines (e.g. :code:`fv3gfs.initialize`), the current
working directory is a valid run directory. This can be done within the model script using the :code:`fv3config`
package (as is done in the provided example scripts), or any other method you'd like to use.

You additionally may need to extend the stack size using `ulimit -s unlimited`. Not doing this may result in a
segmentation fault.


Running the model
-----------------

Basic model operation can be seen in this short, self-explanatory example::

    from mpi4py import MPI
    import fv3gfs

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()
    fv3gfs.cleanup()

This operation can be modified with getters and setters, described below, to read model state or
insert new operations into the model. As of writing you do need to include both :code:`step_dynamics`
and :code:`step_physics` in each iteration, in that order. :code:`fv3gfs.get_step_count` gets the
number of times the Fortran code itself would execute the main loop before exiting.
:code:`fv3gfs.cleanup()` could be ommitted as far as memory cleanup is concerned, but it does write
final restart output and should be included if you want to maintain the normal behavior you would get
in the Fortran model.

Instead of using a Python script, it is also possible to get precisely the behavior of :code:`fv3.exe` using

.. code-block:: shell

    $ python -m fv3gfs.run


Getting/setting state
---------------------

In addition to just running the model the same as in Fortran, the Python wrapper allows you to interact
with the model state while it is still running. This is done through :code:`get_state` and :code:`set_state`.
The getters and setters presently work with arrays that include ghost cells in the horizontal. You can
remove ghost cells from all arrays in a dictionary using :code:`without_ghost_cells`.

The getters will return arrays which nominally have units information, but these aren't guaranteed
to be accurate and in some cases are indicated as missing.

An example which gets and sets the model state to damp moisture is present in the examples folder of this repo.

You should also keep in mind that the arrays used by :code:`get_state` and :code:`set_state` are domain-decomposed
fields (as opposed to global fields). As of writing we do not have logic to go from a local 0-based index to
a global index, but this can be done in principle by retrieving the process rank from :code:`mpi4py`.


.. autofunction:: fv3gfs.get_state

.. autofunction:: fv3gfs.set_state

.. autofunction:: fv3gfs.without_ghost_cells
