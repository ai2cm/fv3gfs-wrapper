import unittest
import os
from copy import deepcopy
import numpy as np
import fv3gfs.wrapper
import fv3gfs.util
from mpi4py import MPI
from util import get_default_config, main


test_dir = os.path.dirname(os.path.abspath(__file__))


def prescribe_sea_surface_temperature_with_random_values():
    old_state = fv3gfs.wrapper.get_state(names=["ocean_surface_temperature"])
    replace_state = deepcopy(old_state)
    for name, quantity in replace_state.items():
        quantity.view[:] = np.random.uniform(size=quantity.extent)
    fv3gfs.wrapper.set_state(replace_state)
    return replace_state


def get_state_single_variable(name):
    return fv3gfs.wrapper.get_state([name])[name].view[:]


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
        prescribe_sea_surface_temperature_with_random_values()
        fv3gfs.wrapper.step()
        air_temperature_from_prescribed_ocean_temperature = get_state_single_variable(
            "air_temperature"
        )

        # We expect these states to differ.
        assert not np.array_equal(
            air_temperature_from_default_ocean_temperature,
            air_temperature_from_prescribed_ocean_temperature,
        )

    def test_prescribing_sst_changes_surface_temperature_diagnostic(self):
        replaced_state = prescribe_sea_surface_temperature_with_random_values()
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
    config["namelist"]["gfs_physics_nml"]["override_sea_surface_temperature"] = True
    main(test_dir, config)
