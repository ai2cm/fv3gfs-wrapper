import unittest
import os
import shutil
from copy import deepcopy
import yaml
import numpy as np
import fv3config
import fv3gfs
import fv3util
from mpi4py import MPI
from util import redirect_stdout

test_dir = os.path.dirname(os.path.abspath(__file__))


class SetterTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SetterTests, self).__init__(*args, **kwargs)
        self.tracer_data = fv3gfs.get_tracer_metadata()
        self.dynamics_data = {
            entry["name"]: entry for entry in fv3util.DYNAMICS_PROPERTIES
        }
        self.physics_data = {
            entry["name"]: entry for entry in fv3util.PHYSICS_PROPERTIES
        }

    def setUp(self):
        pass

    def tearDown(self):
        MPI.COMM_WORLD.barrier()

    def test_dynamics_names_present(self):
        """Test that some small subset of dynamics names are in the data dictionary"""
        for name in ["x_wind", "y_wind", "vertical_wind", "surface_geopotential"]:
            self.assertIn(name, self.dynamics_data.keys())

    def test_physics_names_present(self):
        """Test that some small subset of physics names are in the data dictionary"""
        for name in [
            "land_sea_mask",
            "surface_temperature",
            "surface_roughness",
            "air_temperature_at_2m",
        ]:
            self.assertIn(name, self.physics_data.keys())

    def test_set_then_get_cloud_amount(self):
        """Included because this is the last diagnostic tracer in memory"""
        self._set_names_helper(["cloud_amount"])

    def test_dynamics_names_one_at_a_time(self):
        for name in self.dynamics_data.keys():
            with self.subTest(msg=name):
                self._set_names_helper([name])
                MPI.COMM_WORLD.barrier()

    def test_physics_names_one_at_a_time(self):
        for name in self.physics_data.keys():
            with self.subTest(msg=name):
                self._set_names_helper([name])
                MPI.COMM_WORLD.barrier()

    def test_tracer_names_one_at_a_time(self):
        for name in self.tracer_data.keys():
            self._set_names_helper([name])
            MPI.COMM_WORLD.barrier()

    def test_set_then_get_air_temperature(self):
        self._set_names_helper(["air_temperature"])

    def test_set_then_get_all_dynamics_names(self):
        self._set_names_helper(self.dynamics_data.keys())

    def test_set_then_get_all_physics_names(self):
        self._set_names_helper(self.physics_data.keys())

    def test_set_then_get_all_tracer_names(self):
        self._set_names_helper(self.tracer_data.keys())

    def test_set_then_get_all_names(self):
        self._set_names_helper(
            list(self.dynamics_data.keys())
            + list(self.physics_data.keys())
            + list(self.tracer_data.keys())
        )

    def test_set_then_get_only_some_names(self):
        all_name_list = (
            list(self.dynamics_data.keys())
            + list(self.physics_data.keys())
            + list(self.tracer_data.keys())
        )
        self._set_names_helper(all_name_list[::3])

    def test_set_non_existent_quantity(self):
        with self.assertRaises(ValueError):
            fv3gfs.set_state(
                {
                    "non_quantity": fv3util.Quantity(
                        np.array([0.0]), dims=["dim1"], units=""
                    )
                }
            )

    def _set_names_helper(self, name_list):
        self._set_all_names_at_once_helper(name_list)
        self._set_names_one_at_a_time_helper(name_list)

    def _set_all_names_at_once_helper(self, name_list):
        old_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(old_state, name_list)
        replace_state = deepcopy(old_state)
        for name, quantity in replace_state.items():
            quantity.view[:] = np.random.uniform(size=quantity.extent)
        fv3gfs.set_state(replace_state)
        new_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(new_state, name_list)
        for name, new_quantity in new_state.items():
            with self.subTest(name):
                replace_quantity = replace_state[name]
                self.assert_values_equal(new_quantity, replace_quantity)

    def _set_names_one_at_a_time_helper(self, name_list):
        old_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(old_state, name_list)
        for replace_name in name_list:
            with self.subTest(replace_name):
                quantity = deepcopy(old_state[replace_name])
                quantity.view[:] = np.random.uniform(size=quantity.extent)
                replace_state = {replace_name: quantity}
                fv3gfs.set_state(replace_state)
                new_state = fv3gfs.get_state(names=name_list)
                self._check_gotten_state(new_state, name_list)
                for name, new_quantity in new_state.items():
                    if name == replace_name:
                        self.assertFalse(new_quantity.np.all(new_quantity.view[:] == old_state[name].view[:]), "quantity has not changed")
                        self.assert_values_equal(new_quantity, replace_state[name])
                    else:
                        self.assert_values_equal(new_quantity, old_state[name])
                old_state = new_state

    def _check_gotten_state(self, state, name_list):
        for name, value in state.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, fv3util.Quantity)
        for name in name_list:
            self.assertIn(name, state)
        self.assertEqual(len(name_list), len(state.keys()))

    def assert_values_equal(self, quantity1, quantity2):
        self.assertTrue(quantity1.np.all(quantity1.view[:] == quantity2.view[:]))


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    with open(os.path.join(test_dir, "default_config.yml"), "r") as f:
        config = yaml.safe_load(f)
    rundir = os.path.join(test_dir, "rundir")
    if rank == 0:
        if os.path.isdir(rundir):
            shutil.rmtree(rundir)
        fv3config.write_run_directory(config, rundir)
    MPI.COMM_WORLD.barrier()
    original_path = os.getcwd()
    os.chdir(rundir)
    try:
        with redirect_stdout(os.devnull):
            fv3gfs.initialize()
            MPI.COMM_WORLD.barrier()
        if rank != 0:
            kwargs = {"verbosity": 0}
        else:
            kwargs = {"verbosity": 2}
        unittest.main(**kwargs)
    finally:
        os.chdir(original_path)
        if rank == 0:
            shutil.rmtree(rundir)
