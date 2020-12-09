import fv3gfs.wrapper
import fv3gfs.wrapper.examples

if __name__ == "__main__":
    # initialize the machine learning model
    rf_model = fv3gfs.wrapper.examples.get_random_forest()
    fv3gfs.wrapper.initialize()
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()
        
        # apply an update from the machine learning model
        state = fv3gfs.wrapper.get_state(rf_model.inputs)
        rf_model.update(state)
        fv3gfs.wrapper.set_state(state)
        
        fv3gfs.wrapper.save_intermediate_restart_if_enabled()
    fv3gfs.wrapper.cleanup()
