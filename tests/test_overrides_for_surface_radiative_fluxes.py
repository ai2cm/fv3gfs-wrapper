import unittest
import os
from copy import deepcopy
from fv3gfs.wrapper._properties import OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES
import numpy as np
import fv3gfs.wrapper
import fv3gfs.util
from mpi4py import MPI
from util import main
import yaml

test_dir = os.path.dirname(os.path.abspath(__file__))
(
    DOWNWARD_LONGWAVE,
    DOWNWARD_SHORTWAVE,
    NET_SHORTWAVE,
) = OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES


def override_surface_radiative_fluxes_with_random_values():
    old_state = fv3gfs.wrapper.get_state(names=OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES)
    replace_state = deepcopy(old_state)
    for name, quantity in replace_state.items():
        quantity.view[:] = np.random.uniform(size=quantity.extent)
    fv3gfs.wrapper.set_state(replace_state)
    return replace_state


def get_state_single_variable(name):
    return fv3gfs.wrapper.get_state([name])[name].view[:]


class OverridingSurfaceRadiativeFluxTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OverridingSurfaceRadiativeFluxTests, self).__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        MPI.COMM_WORLD.barrier()

    def test_overriding_fluxes_changes_model_state(self):
        checkpoint_state = fv3gfs.wrapper.get_state(fv3gfs.wrapper.get_restart_names())

        fv3gfs.wrapper.step()
        temperature_with_default_override = get_state_single_variable("air_temperature")

        # Restore state to original checkpoint; modify the radiative fluxes;
        # step the model again.
        fv3gfs.wrapper.set_state(checkpoint_state)
        override_surface_radiative_fluxes_with_random_values()
        fv3gfs.wrapper.step()
        temperature_with_random_override = get_state_single_variable("air_temperature")

        # We expect these states to differ.
        assert not np.array_equal(
            temperature_with_default_override, temperature_with_random_override
        )

    def test_overriding_fluxes_are_propagated_to_diagnostics(self):
        replace_state = override_surface_radiative_fluxes_with_random_values()

        # We need to step the model to fill the diagnostics buckets.
        fv3gfs.wrapper.step()

        timestep = fv3gfs.wrapper.flags.dt_atmos
        expected_DSWRFI = replace_state[DOWNWARD_SHORTWAVE].view[:]
        expected_DLWRFI = replace_state[DOWNWARD_LONGWAVE].view[:]
        expected_USWRFI = (
            replace_state[DOWNWARD_SHORTWAVE].view[:]
            - replace_state[NET_SHORTWAVE].view[:]
        )

        result_DSWRF = fv3gfs.wrapper.get_diagnostic_by_name("DSWRF").view[:]
        result_DLWRF = fv3gfs.wrapper.get_diagnostic_by_name("DLWRF").view[:]
        result_USWRF = fv3gfs.wrapper.get_diagnostic_by_name("USWRF").view[:]
        result_DSWRFI = fv3gfs.wrapper.get_diagnostic_by_name("DSWRFI").view[:]
        result_DLWRFI = fv3gfs.wrapper.get_diagnostic_by_name("DLWRFI").view[:]
        result_USWRFI = fv3gfs.wrapper.get_diagnostic_by_name("USWRFI").view[:]

        np.testing.assert_allclose(result_DSWRF, timestep * expected_DSWRFI)
        np.testing.assert_allclose(result_DLWRF, timestep * expected_DLWRFI)
        np.testing.assert_allclose(result_USWRF, timestep * expected_USWRFI)
        np.testing.assert_allclose(result_DSWRFI, expected_DSWRFI)
        np.testing.assert_allclose(result_DLWRFI, expected_DLWRFI)
        np.testing.assert_allclose(result_USWRFI, expected_USWRFI)


if __name__ == "__main__":
    with open(os.path.join(test_dir, "default_config.yml"), "r") as f:
        config = yaml.safe_load(f)
    config["namelist"]["gfs_physics_nml"]["override_surface_radiative_fluxes"] = True
    main(test_dir, config)
