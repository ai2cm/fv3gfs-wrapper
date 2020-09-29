import unittest
import yaml
import os
import shutil
import numpy as np
import fv3config
import fv3gfs.wrapper
import fv3gfs.util
from mpi4py import MPI

test_dir = os.path.dirname(os.path.abspath(__file__))
MM_PER_M = 1000


class GetterTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GetterTests, self).__init__(*args, **kwargs)
        self.mpi_comm = MPI.COMM_WORLD

    def setUp(self):
        pass

    def tearDown(self):
        self.mpi_comm.barrier()

    def test_get_diag_info(self):
        output = fv3gfs.wrapper.get_diagnostic_info()
        for item in output:
            self.assertIsInstance(item["axes"], int)
            self.assertIsInstance(item["mod_name"], str)
            self.assertIsInstance(item["name"], str)
            self.assertIsInstance(item["desc"], str)
            self.assertIsInstance(item["unit"], str)
            
