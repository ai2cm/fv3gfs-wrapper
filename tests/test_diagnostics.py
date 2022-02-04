import unittest
import os
import fv3gfs.wrapper
import pace.util
from mpi4py import MPI

from util import get_default_config, main

test_dir = os.path.dirname(os.path.abspath(__file__))


class DiagnosticTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpi_comm = MPI.COMM_WORLD

    def setUp(self):
        pass

    def tearDown(self):
        self.mpi_comm.barrier()

    def test_get_diag_info(self):
        output = fv3gfs.wrapper._get_diagnostic_info()
        assert len(output) > 0
        for index, item in output.items():
            self.assertIsInstance(item.axes, int)
            self.assertIsInstance(item.module_name, str)

            self.assertIsInstance(item.name, str)
            self.assertIsNot(item.name, "")

            self.assertIsInstance(item.description, str)
            self.assertIsInstance(item.unit, str)

    def test_get_diagnostic_data(self):
        names_to_get = ["tendency_of_air_temperature_due_to_microphysics", "ulwsfc"]
        for name in names_to_get:
            quantity = fv3gfs.wrapper.get_diagnostic_by_name(
                name, module_name="gfs_phys"
            )
            info = fv3gfs.wrapper.get_diagnostic_metadata_by_name(
                name, module_name="gfs_phys"
            )
            self.assertIsInstance(quantity, pace.util.Quantity)
            assert quantity.view[:].ndim == info.axes
            assert quantity.units == info.unit


if __name__ == "__main__":
    config = get_default_config()
    main(test_dir, config)
