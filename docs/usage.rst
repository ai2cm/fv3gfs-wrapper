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

You also need to ensure that when the script executes model subroutines (e.g. :py:func:`fv3gfs.initialize`), the current
working directory is a valid run directory. This can be done within the model script using the :code:`fv3config`
package (as is done in the provided example scripts), or any other method you'd like to use.

You additionally may need to extend the stack size using `ulimit -s unlimited`. Not doing this may result in a
segmentation fault. The docker image is set up to use this setting by default in bash.

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
insert new operations into the model. As of writing you do need to include both :py:func:`fv3gfs.step_dynamics`
and :py:func:`fv3gfs.step_physics` in each iteration, in that order. :py:func:`fv3gfs.get_step_count` gets the
number of times the Fortran code itself would execute the main loop before exiting.
:py:func:`fv3gfs.cleanup` could be ommitted as far as memory cleanup is concerned, but it does write
final restart output and should be included if you want to maintain the normal behavior you would get
in the Fortran model.

.. note::
    The :py:func:`fv3gfs.step_physics` call can be separated into two calls, :py:func:`fv3gfs.compute_physics`
    and :py:func:`fv3gfs.apply_physics` to allow modification of the model state in between these operations.
    However care should be taken when doing this, because the physics and dynamics states are not
    consistent with each other before :py:func:`fv3gfs.apply_physics` is called.

Instead of using a Python script, it is also possible to get precisely the behavior of :code:`fv3.exe` using

.. code-block:: shell

    $ python -m fv3gfs.run

Nudging
-------

Nudging functionality is provided by :py:func:`fv3gfs.apply_nudging` and
:py:func:`fv3gfs.get_nudging_tendencies`. The nudging tendencies can be stored to disk
by the user, for example using a :py:class:`fv3gfs.ZarrMonitor`. A runfile using this
functionality can be found in the `examples` directory.

Diagnostic IO
-------------

State can be persisted to disk using either :py:func:`fv3gfs.write_state` (described below)
or :py:class:`fv3gfs.ZarrMonitor`. The latter will coordinate between ranks to
write state to a unified Zarr store. Initializing it requires passing grid information.
This can be done directly from the namelist in a configuration dictionary like so::

    import fv3gfs
    from mpi4py import MPI
    import yaml

    with open('fv3config.yml', 'r') as f:
        config = yaml.safe_load(f)
    partitioner = fv3gfs.TilePartitioner.from_namelist(config['namelist'])

Alternatively, the grid information can be specified manually::

    partitioner = fv3gfs.TilePartitioner(
        layout=(1, 1)
    )

Once you have a :py:class:`fv3gfs.Partitioner`, the monitor can be created using any
Zarr store::

    import zarr
    store = zarr.storage.DirectoryStore('output_dir')  # relative or absolute path
    ZarrMonitor(partitioner, store, mode='w', mpi_comm=MPI.COMM_WORLD)

Note this can be used with any directory store available in ``zarr``.

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
the model state after the Fortran initialization routines take place. Each MPI rank
(process) reads (with :py:func:`fv3gfs.read_state`) or writes (with :py:func:`fv3gfs.write_state)
a netCDF file with all of its restart data. :py:func:`fv3gfs.get_restart_names` returns
a list of all quantity names required to restart the model.

:py:func:`save_intermediate_restart_if_enabled`
will call the portion of the normal Fortran main loop that checks how many timesteps have elapsed
since the last restart was written, and writes out restart files with the model time stamp
if intermediate restarts are enabled in the namelist and the correct number of timesteps
have elapsed. :py:func:`save_fortran_restart` will immediately save restart files with the
given label (instead of the model timestamp). :py:func:`load_fortran_restart_folder`
will load restart files from the given directory, using the provided label if given (e.g. timestamp
if Fortran intermediate restarts, or chosen saved label if using the wrapper direct-save routine).

Loading legacy restarts
-----------------------

A function :py:func:`fv3gfs.open_restart` is available to load restart files that have
been output by the Fortran code. This routine will handle
loading the data on a single processor per tile and then distribute the data to other
processes on the same tile. This may cause out-of-memory errors, which can be mitigated
in a couple different ways through changes to the code base (e.g. loading a subset of
the variables or levels at a time before distributing across ranks).

