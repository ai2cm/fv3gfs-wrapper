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
DOWNWARD_LONGWAVE = "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface"
DOWNWARD_SHORTWAVE = "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface"
NET_SHORTWAVE = "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface"


class OverridingFluxDiagnosticsTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OverridingFluxDiagnosticsTests, self).__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        MPI.COMM_WORLD.barrier()

    def test_overriding_fluxes_are_propagated_diagnostics(self):
        old_state = fv3gfs.wrapper.get_state(names=OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES)
        replace_state = deepcopy(old_state)
        for name, quantity in replace_state.items():
            quantity.view[:] = np.random.uniform(size=quantity.extent)
        fv3gfs.wrapper.set_state(replace_state)

        # We need to step the model to fill the diagnostics buckets.
        fv3gfs.wrapper.step()

        expected_DSWRFI = replace_state[DOWNWARD_SHORTWAVE].view[:]
        expected_DLWRFI = replace_state[DOWNWARD_LONGWAVE].view[:]
        expected_USWRFI = (replace_state[DOWNWARD_SHORTWAVE].view[:]
            - replace_state[NET_SHORTWAVE].view[:]
        )

        result_DSWRFI = fv3gfs.wrapper.get_diagnostic_by_name("DSWRFI").view[:]
        result_DLWRFI = fv3gfs.wrapper.get_diagnostic_by_name("DLWRFI").view[:]
        result_USWRFI = fv3gfs.wrapper.get_diagnostic_by_name("USWRFI").view[:]

        np.testing.assert_allclose(result_DSWRFI, expected_DSWRFI)
        np.testing.assert_allclose(result_DLWRFI, expected_DLWRFI)
        np.testing.assert_allclose(result_USWRFI, expected_USWRFI)


if __name__ == "__main__":
    with open(os.path.join(test_dir, "default_config.yml"), "r") as f:
        config = yaml.safe_load(f)
    config["namelist"]["gfs_physics_nml"]["override_surface_radiative_fluxes"] = True
    main(test_dir, config)
