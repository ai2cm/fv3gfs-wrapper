import unittest
import json
import os
import shutil
from copy import deepcopy
import xarray as xr
import numpy as np
import fv3config
import fv3gfs
from mpi4py import MPI
from util import redirect_stdout

test_dir = os.path.dirname(os.path.abspath(__file__))


class SetterTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SetterTests, self).__init__(*args, **kwargs)
        self.tracer_data = fv3gfs.get_tracer_metadata()
        self.dynamics_data = {
            entry['name']: entry for entry in json.load(open(
                os.path.join(test_dir, '../fv3gfs/dynamics_properties.json')
            ))
        }
        self.physics_data = {
            entry['name']: entry for entry in json.load(open(
                os.path.join(test_dir, '../fv3gfs/physics_properties.json')
            ))
        }

    def setUp(self):
        pass

    def tearDown(self):
        MPI.COMM_WORLD.barrier()

    def test_dynamics_names_present(self):
        """Test that some small subset of dynamics names are in the data dictionary"""
        for name in [
                'x_wind', 'y_wind', 'vertical_wind',
                'surface_geopotential']:
            self.assertIn(name, self.dynamics_data.keys())

    def test_physics_names_present(self):
        """Test that some small subset of physics names are in the data dictionary"""
        for name in [
                'land_sea_mask', 'surface_temperature', 'surface_roughness',
                'air_temperature_at_2m']:
            self.assertIn(name, self.physics_data.keys())

    def test_set_then_get_cloud_amount(self):
        """Included because this is the last diagnostic tracer in memory"""
        self._set_names_helper(['cloud_amount'])

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
        self._set_names_helper(['air_temperature'])

    def test_set_then_get_all_dynamics_names(self):
        self._set_names_helper(self.dynamics_data.keys())

    def test_set_then_get_all_physics_names(self):
        self._set_names_helper(self.physics_data.keys())

    def test_set_then_get_all_tracer_names(self):
        self._set_names_helper(self.tracer_data.keys())

    def test_set_then_get_all_names(self):
        self._set_names_helper(
            list(self.dynamics_data.keys()) +
            list(self.physics_data.keys()) +
            list(self.tracer_data.keys())
        )

    def test_set_then_get_only_some_names(self):
        all_name_list = (list(
            self.dynamics_data.keys()) +
            list(self.physics_data.keys()) +
            list(self.tracer_data.keys())
        )
        self._set_names_helper(all_name_list[::3])

    def test_set_non_existent_quantity(self):
        with self.assertRaises(ValueError):
            fv3gfs.set_state({
                'non_quantity': xr.DataArray([0.], dims=['dim1'], attrs={'units': ''})
            })

    def _set_names_helper(self, name_list):
        self._set_all_names_at_once_helper(name_list)
        self._set_names_one_at_a_time_helper(name_list)

    def _set_all_names_at_once_helper(self, name_list):
        old_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(old_state, name_list)
        replace_state = deepcopy(old_state)
        for name, data_array in replace_state.items():
            data_array.values[:] = np.random.uniform(size=data_array.shape)
        fv3gfs.set_state(replace_state)
        new_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(new_state, name_list)
        for name, new_data_array in new_state.items():
            with self.subTest(name):
                replace_data_array = replace_state[name]
                self.assert_values_equal(new_data_array, replace_data_array)

    def _set_names_one_at_a_time_helper(self, name_list):
        old_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(old_state, name_list)
        for replace_name in name_list:
            with self.subTest(replace_name):
                data_array = deepcopy(old_state[replace_name])
                data_array.values[:] = np.random.uniform(size=data_array.shape)
                replace_state = {
                    replace_name: data_array
                }
                fv3gfs.set_state(replace_state)
                new_state = fv3gfs.get_state(names=name_list)
                self._check_gotten_state(new_state, name_list)
                for name, new_data_array in new_state.items():
                    if name == replace_name:
                        self.assert_values_equal(new_data_array, replace_state[name])
                    else:
                        self.assert_values_equal(new_data_array, old_state[name])
                old_state = new_state

    def _check_gotten_state(self, state, name_list):
        for name, value in state.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, xr.DataArray)
            self.assertIn('units', value.attrs)
        for name in name_list:
            self.assertIn(name, state)
        self.assertEqual(len(name_list), len(state.keys()))

    def assert_values_equal(self, data_array1, data_array2):
        self.assertTrue(np.all(data_array1.values == data_array2.values))


if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    config = fv3config.get_default_config()
    rundir = os.path.join(test_dir, 'rundir')
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
            kwargs = {'verbosity': 0}
        else:
            kwargs = {'verbosity': 2}
        unittest.main(**kwargs)
    finally:
        os.chdir(original_path)
        if rank == 0:
            shutil.rmtree(rundir)
