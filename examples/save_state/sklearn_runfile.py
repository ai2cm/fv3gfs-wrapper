import os
from datetime import timedelta
from mpi4py import MPI
import fv3gfs
import fv3config
import pickle
import fsspec
import xarray as xr
import state_io
import run_sklearn
import logging
import f90nml
import zarr

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('python')


# This code shows an example where we relax specific humidity towards zero with a 7-day timescale.
# The relaxation is done purely in Python. You can use a similar method for any kind of online
# model operations, whether it's an online machine learning code or saving online diagnostic output.

# May need to run 'ulimit -s unlimited' before running this example
# If you're running in our prepared docker container, you definitely need to do this
# sets the stack size to unlimited

# Run using mpirun -n 6 python3 basic_model.py
# mpirun flags that may be useful:
#     for docker:  --allow-run-as-root
#     for CircleCI: --oversubscribe
#     to silence a certain inconsequential MPI error: --mca btl_vader_single_copy_mechanism none

# All together:
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 save_state_runfile.py

DELP = "pressure_thickness_of_atmospheric_layer"
MODEL = run_sklearn.open_sklearn_model(run_sklearn.SKLEARN_MODEL)
VARIABLES = list(state_io.CF_TO_RESTART_MAP)+ [DELP]


rundir_basename = 'rundir'
input_nml = 'rundir/input.nml'
NML = f90nml.read(input_nml)
TIMESTEP = NML['coupler_nml']['dt_atmos']


if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    group = zarr.open_group('test.zarr', mode='w')
    q2_writer = state_io.ZarrVariableWriter(comm, group, name='Q2')
    net_precip_writer = state_io.ZarrVariableWriter(comm, group, name='net_precip') 

    rank = comm.Get_rank()
    current_dir = os.getcwd()
    rundir_path = os.path.join(current_dir, rundir_basename)
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    os.chdir(rundir_path)
    if rank == 0:
        logger.info(f"Timestep: {TIMESTEP}")

    # Calculate factor for relaxing humidity to zero
    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()

 
        if rank == 0:
            logger.debug(f"Getting state variables: {VARIABLES}")
        state = fv3gfs.get_state(names=VARIABLES)
        if rank == 0:
            logger.debug("Computing RF updated variables")
        preds, diags = run_sklearn.update(MODEL, state, dt=TIMESTEP)
        if rank == 0:
            logger.debug("Setting Fortran State")
        fv3gfs.set_state(preds)
        if rank == 0:
            logger.debug("Setting Fortran State")

        q2_writer.append(diags['Q2'])
        net_precip = (diags['Q2'] * state[DELP] / 9.81).sum('z')
        net_precip_writer.append(net_precip)
        

    fv3gfs.cleanup()
