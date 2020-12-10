import fv3gfs.wrapper
import fv3gfs.wrapper.examples
import f90nml
from datetime import timedelta

if __name__ == "__main__":
    # load timestep from the namelist
    namelist = f90nml.read("input.nml")
    timestep = timedelta(seconds=namelist["coupler_nml"]["dt_atmos"])
    # initialize the machine learning model
    rf_model = fv3gfs.wrapper.examples.get_random_forest()
    fv3gfs.wrapper.initialize()
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()

        # apply an update from the machine learning model
        state = fv3gfs.wrapper.get_state(rf_model.inputs)
        rf_model.update(state, timestep=timestep)
        fv3gfs.wrapper.set_state(state)

        fv3gfs.wrapper.save_intermediate_restart_if_enabled()
    fv3gfs.wrapper.cleanup()
