import fv3gfs.wrapper
import numpy as np
from mpi4py import MPI

ROOT = 0

if __name__ == "__main__":
    fv3gfs.wrapper.initialize()
    # MPI4py requires a receive "buffer" array to store incoming data
    min_surface_temperature = np.array(0.0)
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()

        # Retrieve model minimum surface temperature
        state = fv3gfs.wrapper.get_state(["surface_temperature"])
        MPI.COMM_WORLD.Reduce(
            state["surface_temperature"].view[:].min(),
            min_surface_temperature,
            root=ROOT,
            op=MPI.MIN,
        )
        if MPI.COMM_WORLD.Get_rank() == ROOT:
            units = state["surface_temperature"].units
            print(f"Minimum surface temperature: {min_surface_temperature} {units}")

        fv3gfs.wrapper.save_intermediate_restart_if_enabled()
    fv3gfs.wrapper.cleanup()
