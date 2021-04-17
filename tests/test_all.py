import unittest
from mpi4py import MPI
from util import run_unittest_script


# The packages we import will import MPI, causing an MPI init, but we don't actually
# want to use MPI under this script. We have to finalize so mpirun will work on
# the test scripts we call that *do* need MPI.
MPI.Finalize()


class UsingMPITests(unittest.TestCase):
    def test_getters(self):
        run_unittest_script("test_getters.py")

    def test_setters_default(self):
        run_unittest_script("test_setters.py", "false")

    def test_setters_while_overriding_surface_radiative_fluxes(self):
        run_unittest_script("test_setters.py", "true")

    def test_overrides_for_surface_radiative_fluxes_modify_diagnostics(self):
        run_unittest_script("test_overrides_for_surface_radiative_fluxes.py")

    def test_diagnostics(self):
        run_unittest_script("test_diagnostics.py")

    def test_tracer_metadata(self):
        run_unittest_script("test_tracer_metadata.py")

    def test_get_time_julian(self):
        run_unittest_script("test_get_time.py", "julian")

    def test_get_time_thirty_day(self):
        run_unittest_script("test_get_time.py", "thirty_day")

    def test_get_time_noleap(self):
        run_unittest_script("test_get_time.py", "noleap")

    def test_flags(self):
        run_unittest_script("test_flags.py")

    def test_set_ocean_surface_temperature(self):
        run_unittest_script("test_set_ocean_surface_temperature.py")


if __name__ == "__main__":
    unittest.main()
