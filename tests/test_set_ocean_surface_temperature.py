import unittest
import os
import numpy as np
import fv3gfs.wrapper
import fv3gfs.util
from mpi4py import MPI
from util import (
    get_default_config,
    get_state_single_variable,
    main,
    replace_state_with_random_values,
)


test_dir = os.path.dirname(os.path.abspath(__file__))


def mask_non_ocean_values(field):
    is_ocean = np.isclose(get_state_single_variable("land_sea_mask"), 0.0)
    return np.where(is_ocean, field, np.nan)


class PrescribeSSTTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PrescribeSSTTests, self).__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        MPI.COMM_WORLD.barrier()

    def test_prescribing_sst_changes_model_state(self):
        checkpoint_state = fv3gfs.wrapper.get_state(fv3gfs.wrapper.get_restart_names())

        fv3gfs.wrapper.step()
        air_temperature_from_default_ocean_temperature = get_state_single_variable(
            "air_temperature"
        )

        # Restore state to original checkpoint; modify the SST;
        # step the model again.
        fv3gfs.wrapper.set_state(checkpoint_state)
        replace_state_with_random_values(["ocean_surface_temperature"])
        fv3gfs.wrapper.step()
        air_temperature_from_prescribed_ocean_temperature = get_state_single_variable(
            "air_temperature"
        )

        # We expect these states to differ.
        assert not np.allclose(
            air_temperature_from_default_ocean_temperature,
            air_temperature_from_prescribed_ocean_temperature,
        )

    def test_prescribing_sst_changes_surface_temperature_diagnostic(self):
        replaced_state = replace_state_with_random_values(["ocean_surface_temperature"])
        prescribed_sst = replaced_state["ocean_surface_temperature"].view[:]
        fv3gfs.wrapper.step()
        surface_temperature_diagnostic = fv3gfs.wrapper.get_diagnostic_by_name(
            "tsfc", module_name="gfs_sfc"
        ).view[:]

        # Over the ocean we expect these results to be equal.
        result = mask_non_ocean_values(surface_temperature_diagnostic)
        expected = mask_non_ocean_values(prescribed_sst)
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    config = get_default_config()
    config["namelist"]["gfs_physics_nml"]["use_climatological_sst"] = False
    main(test_dir, config)
