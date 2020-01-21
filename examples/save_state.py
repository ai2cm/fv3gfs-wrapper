import os
from datetime import timedelta
from mpi4py import MPI
import fv3gfs
import fv3config
import pickle
import fsspec
import xarray as xr
import state_io

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
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 save_state.py

def get_names(props):
    for var in props:
        yield var['name']

VARIABLES = list(get_names(fv3gfs.dynamics_properties)) + list(get_names(fv3gfs.physics_properties))


rundir_basename = 'rundir'
output_path = 'state.pkl'

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    current_dir = os.getcwd()
    rundir_path = os.path.join(current_dir, rundir_basename)
    config = fv3config.get_default_config()
    if rank == 0:  # Only create run directory from one rank
        # Can alter this config dictionary to configure the run
        fv3config.write_run_directory(config, rundir_path)
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    os.chdir(rundir_path)

    # Calculate factor for relaxing humidity to zero
    relaxation_rate = timedelta(days=7)
    timestep = timedelta(seconds=config['namelist']['coupler_nml']['dt_atmos'])

    fv3gfs.initialize()
    fv3gfs.step_dynamics()
    fv3gfs.step_physics()
    state = fv3gfs.get_state(names=VARIABLES)
    combined = comm.gather(state, root=0)

    if rank == 0:
        with fsspec.open(output_path, "wb") as f:
            state_io.dump(combined, f)
    fv3gfs.cleanup()
