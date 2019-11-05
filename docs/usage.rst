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
The getters and setters do not include ghost cells in the horizontal. If you need to add them, you can add and
remove ghost cells using :code:`with_ghost_cells` and :code:`without_ghost_cells`.
The getters will return arrays which nominally have units information, but these aren't guaranteed
to be accurate and in some cases are indicated as missing.

An example which gets and sets the model state to damp moisture is present in the examples folder of this repo.

You should also keep in mind that the arrays used by :code:`get_state` and :code:`set_state` are domain-decomposed
fields (as opposed to global fields). As of writing we do not have logic to go from a local 0-based index to
a global index, but this can be done in principle by retrieving the process rank from :code:`mpi4py`.

.. autofunction:: fv3gfs.get_state

.. autofunction:: fv3gfs.set_state

.. autofunction:: fv3gfs.with_ghost_cells

.. autofunction:: fv3gfs.without_ghost_cells


Restart
-------

Sometimes you may want to write out model state to disk so that you can restart the model
from this state later. The FV3GFS Fortran model provides functionality to do so -- we will describe
functions to interface with these Fortran restarts further below.

As a replacement, we provide a python-centric method for saving out and loading model state.
Earlier we described :py:func:`fv3gfs.get_state`, which takes in a list of names of quantities to retrieve
from the Fortran state. Also provided is `get_restart_names`, which returns a list of quantity
names you would need to write out to disk in order to smoothly reset the model state to that point.

For example, if you ran::

    checkpoint_state = fv3gfs.get_state(fv3gfs.get_restart_names())
    [time steps, model operations, etc.]
    fv3gfs.set_state(checkpoint_state)

after calling :py:func:`fv3gfs.set_state`, the model would be reset to the point
where the checkpoint state was retrieved.

The remaining step for restarting from disk is to be able to write model states to/from disk.
For this, we have :py:func:`fv3gfs.write_state` and :py:func:`read_state`. Consider a model
script with a general structure as follows:

.. code-block:: python

    from mpi4py import MPI
    import fv3gfs
    import os

    fv3gfs.initialize()
    restart_filename = os.path.join(
        os.getcwd(),
        f'RESTART/restart.rank{MPI.COMM_WORLD.Get_rank()}.nc'
    )
    if os.path.isfile(restart_filename):
        restart_state = fv3gfs.read_state(restart_filename)
        fv3gfs.set_state(restart_state)

    # ... continue to main loop and other parts of run script

    # after main loop is finished:
    restart_state = fv3gfs.get_state(fv3gfs.get_restart_names())
    fv3gfs.write_state(restart_state, restart_filename)

In this script, if a restart file exists in the RESTART directory, it will be read in and overwrite
the model state after the Fortran initialization routines take place. Each MPI rank (process) reads
or writes a netCDF file with all of its restart data.

:py:func:`save_intermediate_restart_if_enabled`
will call the portion of the normal Fortran main loop that checks how many timesteps have elapsed
since the last restart was written, and writes out restart files with the model time stamp
if intermediate restarts are enabled in the namelist and the correct number of timesteps
have elapsed. :py:func:`save_fortran_restart` will immediately save restart files with the
given label (instead of the model timestamp). :py:func:`load_fortran_restart_folder`
will load restart files from the given directory, using the provided label if given (e.g. timestamp
if Fortran intermediate restarts, or chosen saved label if using the wrapper direct-save routine).

.. autofunction:: fv3gfs.get_restart_names

.. autofunction:: fv3gfs.write_state

.. autofunction:: fv3gfs.read_state
