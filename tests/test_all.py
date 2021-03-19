import unittest
from mpi4py import MPI
from util import run_unittest_script
import os

base_dir = os.path.dirname(os.path.realpath(__file__))


def _run_unittest_script(script_name, *args):
    path = os.path.join(base_dir, script_name)
    run_unittest_script(path, *args)


# The packages we import will import MPI, causing an MPI init, but we don't actually
# want to use MPI under this script. We have to finalize so mpirun will work on
# the test scripts we call that *do* need MPI.
MPI.Finalize()


class UsingMPITests(unittest.TestCase):
    def test_getters(self):
        _run_unittest_script("test_getters.py")

    def test_setters_default(self):
        _run_unittest_script("test_setters.py", "False")

    def test_setters_while_overriding_surface_radiative_fluxes(self):
        _run_unittest_script("test_setters.py", "True")

    def test_overrides_for_surface_radiative_fluxes_modify_diagnostics(self):
        _run_unittest_script("test_overrides_for_surface_fluxes_diagnostics.py")

    def test_diagnostics(self):
        _run_unittest_script("test_diagnostics.py")

    def test_tracer_metadata(self):
        _run_unittest_script("test_tracer_metadata.py")

    def test_get_time_julian(self):
        _run_unittest_script("test_get_time.py", "julian")

    def test_get_time_thirty_day(self):
        _run_unittest_script("test_get_time.py", "thirty_day")

    def test_get_time_noleap(self):
        _run_unittest_script("test_get_time.py", "noleap")

    def test_flags(self):
        _run_unittest_script("test_flags.py")


if __name__ == "__main__":
    unittest.main()
