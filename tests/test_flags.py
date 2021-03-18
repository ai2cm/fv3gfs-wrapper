import unittest
import os
import fv3gfs.wrapper
import fv3gfs.util
from fv3gfs.wrapper._properties import FLAGSTRUCT_PROPERTIES, GFS_CONTROL_PROPERTIES
from mpi4py import MPI

from util import main

test_dir = os.path.dirname(os.path.abspath(__file__))
FORTRAN_TO_PYTHON_TYPE = {"integer": int, "real": float, "logical": bool}


class FlagsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FlagsTest, self).__init__(*args, **kwargs)
        self.flagstruct_data = {entry["name"]: entry for entry in FLAGSTRUCT_PROPERTIES}
        self.gfs_control_data = {
            entry["name"]: entry for entry in GFS_CONTROL_PROPERTIES
        }
        self.mpi_comm = MPI.COMM_WORLD

    def setUp(self):
        pass

    def tearDown(self):
        self.mpi_comm.barrier()

    def test_flagstruct_properties_present_in_metadata(self):
        """Test that some small subset of flagstruct names are in the data dictionary"""
        for name in ["do_adiabatic_init", "n_split"]:
            self.assertIn(name, self.flagstruct_data.keys())

    def test_gfs_control_properties_present_in_metadata(self):
        """Test that some small subset of gfs_control names are in the data dictionary"""
        for name in ["override_surface_radiative_fluxes"]:
            self.assertIn(name, self.gfs_control_data.keys())

    def test_get_all_flagstruct_properties(self):
        self._get_all_properties_helper(self.flagstruct_data)

    def test_get_all_gfs_control_properties(self):
        self._get_all_properties_helper(self.gfs_control_data)

    def _get_all_properties_helper(self, properties):
        for name, data in properties.items():
            with self.subTest(name):
                result = getattr(fv3gfs.wrapper.flags, name)
                expected_type = FORTRAN_TO_PYTHON_TYPE[data["type_fortran"]]
                self.assertIsInstance(result, expected_type)

    def test_override_surface_radiative_fluxes(self):
        """Test that getting a boolean flag produces its expected result."""
        result = fv3gfs.wrapper.flags.override_surface_radiative_fluxes
        self.assertFalse(result)


if __name__ == "__main__":
    main(test_dir)
