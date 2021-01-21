from datetime import timedelta
from mpi4py import MPI
import fv3gfs.wrapper
import yaml

# This code shows an example where we relax specific humidity towards zero with a 7-day timescale.
# The relaxation is done purely in Python. You can use a similar method for any kind of online
# model operations, whether it's an online machine learning code or saving online diagnostic output.

# May need to run 'ulimit -s unlimited' before running this example
# If you're running in our prepared docker container, you definitely need to do this

# Run using mpirun -n 6 python3 basic_model.py
# mpirun flags that may be useful if using openmpi rather than mpich:
#     for docker:  --allow-run-as-root
#     for CircleCI: --oversubscribe
#     to silence a certain inconsequential MPI error: --mca btl_vader_single_copy_mechanism none

# All together for openmpi:
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 online_code.py

rundir_basename = "rundir"

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:  # only use filesystem on one rank
        with open("fv3config.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
        config = comm.bcast(config)
    else:
        config = comm.bcast(None)

    # Calculate factor for relaxing humidity to zero
    relaxation_rate = timedelta(days=7)
    timestep = timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])

    fv3gfs.wrapper.initialize()
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()
        fv3gfs.wrapper.save_intermediate_restart_if_enabled()

        # dry out the model with the given relaxation rate
        state = fv3gfs.wrapper.get_state(names=["specific_humidity"])
        state["specific_humidity"].view[:] -= (
            state["specific_humidity"].view[:]
            * timestep.total_seconds()
            / relaxation_rate.total_seconds()
        )
        fv3gfs.wrapper.set_state(state)
    fv3gfs.wrapper.cleanup()
