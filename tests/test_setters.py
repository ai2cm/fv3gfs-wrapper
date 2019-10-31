import unittest
import json
import os
import sys
import shutil
from contextlib import contextmanager
import ctypes
import io
import tempfile
from copy import deepcopy
import xarray as xr
import numpy as np
import fv3config
import fv3gfs
from mpi4py import MPI

test_dir = os.path.dirname(os.path.abspath(__file__))

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


class redirect_stdout(object):

    def __init__(self, filename):
        self.stream = open(filename, 'wb')
        self._filename = filename
        self._stdout_file_descriptor = sys.stdout.fileno()
        self._saved_stdout_file_descriptor = os.dup(self._stdout_file_descriptor)
        self._temporary_stdout = tempfile.TemporaryFile(mode='w+b')

    def __enter__(self):
        # Redirect the stdout file descriptor to point at our temporary file
        self._redirect_stdout(self._temporary_stdout.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        # Set the stdout file descriptor back to what it was when we started
        self._redirect_stdout(self._saved_stdout_file_descriptor)
        # Write contents of temporary file to the output file
        self.temporary_stdout.flush()
        self.temporary_stdout.seek(0, io.SEEK_SET)
        self.stream.write(self.temporary_stdout.read())
        # Close our temporary file and remove the duplicate file descriptor
        self._temporary_stdout.close()
        os.close(self._saved_stdout_file_descriptor)

    def _redirect_stdout(self, to_file_descriptor):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make self._stdout_file_descriptor point to the same file as to_file_descriptor
        # This redirects stdout to the file at the file descriptor level, which C/Fortran also obeys
        os.dup2(to_file_descriptor, self._stdout_file_descriptor)
        # Create a new sys.stdout for Python that points to the redirected file descriptor
        sys.stdout = io.TextIOWrapper(os.fdopen(self._stdout_file_descriptor, 'wb'))


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
                'eastward_wind', 'northward_wind', 'vertical_wind',
                'surface_geopotential']:
            self.assertIn(name, self.dynamics_data.keys())

    def test_physics_names_present(self):
        """Test that some small subset of physics names are in the data dictionary"""
        for name in [
                'land_sea_mask', 'surface_temperature', 'surface_roughness',
                'air_temperature_at_2m']:
            self.assertIn(name, self.physics_data.keys())

    def test_set_then_get_cloud_amount(self):
        """Included because this caused a segfault at some point, as a diagnostic tracer."""
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
            replace_data_array = replace_state[name]
            self.assert_non_ghost_cells_equal(new_data_array, replace_data_array)

    def _set_names_one_at_a_time_helper(self, name_list):
        old_state = fv3gfs.get_state(names=name_list)
        self._check_gotten_state(old_state, name_list)
        for replace_name in name_list:
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
                    self.assert_non_ghost_cells_equal(new_data_array, replace_state[name])
                else:
                    self.assert_non_ghost_cells_equal(new_data_array, old_state[name])
            old_state = new_state

    def _check_gotten_state(self, state, name_list):
        for name, value in state.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, xr.DataArray)
            self.assertIn('units', value.attrs)
        for name in name_list:
            self.assertIn(name, state)
        self.assertEqual(len(name_list), len(state.keys()))

    def assert_non_ghost_cells_equal(self, data_array1, data_array2):
        data = fv3gfs.without_ghost_cells({
            'array1': data_array1,
            'array2': data_array2
        })
        self.assertTrue(np.all(data['array1'].values == data['array2'].values))


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
        with redirect_stdout(os.path.join(test_dir, f'logs/test_setters.rank{rank}.log')):
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
