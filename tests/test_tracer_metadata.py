import unittest
import os
import shutil
import yaml
from mpi4py import MPI
import fv3config
import fv3gfs.wrapper
from util import redirect_stdout

test_dir = os.path.dirname(os.path.abspath(__file__))
rundir = os.path.join(test_dir, "rundir")


class TracerMetadataTests(unittest.TestCase):
    def test_tracer_index_is_one_based(self):
        data = fv3gfs.wrapper.get_tracer_metadata()
        indexes = []
        for entry in data.values():
            self.assertIn("i_tracer", entry)
            indexes.append(entry["i_tracer"])
        indexes = sorted(indexes)
        self.assertEqual(indexes[0], 1)
        self.assertEqual(indexes[-1], len(indexes))
        self.assertEqual(
            len(indexes), len(set(indexes))
        )  # test there are no duplicates

    def test_tracer_metadata_has_all_keys(self):
        data = fv3gfs.wrapper.get_tracer_metadata()
        for name, metadata in data.items():
            with self.subTest(msg=name):
                self.assertIn("units", metadata)
                self.assertIn("i_tracer", metadata)
                self.assertIn("fortran_name", metadata)
                self.assertIn("restart_name", metadata)
                self.assertIsInstance(metadata["units"], str)
                self.assertIsInstance(metadata["i_tracer"], int)
                self.assertIsInstance(metadata["fortran_name"], str)
                self.assertIsInstance(metadata["restart_name"], str)

    def test_all_tracers_present(self):
        tracer_names = [
            "specific_humidity",
            "cloud_water_mixing_ratio",
            "rain_mixing_ratio",
            "cloud_ice_mixing_ratio",
            "snow_mixing_ratio",
            "graupel_mixing_ratio",
            "ozone_mixing_ratio",
            "cloud_amount",
        ]
        data = fv3gfs.wrapper.get_tracer_metadata()
        self.assertEqual(set(data.keys()), set(tracer_names))

    def test_all_tracers_in_restart_names(self):
        tracer_names = [
            "specific_humidity",
            "cloud_water_mixing_ratio",
            "rain_mixing_ratio",
            "cloud_ice_mixing_ratio",
            "snow_mixing_ratio",
            "graupel_mixing_ratio",
            "ozone_mixing_ratio",
            "cloud_amount",
        ]
        restart_names = fv3gfs.wrapper.get_restart_names()
        missing_names = set(tracer_names).difference(restart_names)
        self.assertEqual(len(missing_names), 0)


if __name__ == "__main__":
    with open(os.path.join(test_dir, "default_config.yml"), "r") as f:
        config = yaml.safe_load(f)
    config[
        "initial_conditions"
    ] = "gs://vcm-fv3config/data/initial_conditions/c12_restart_initial_conditions/v1.0"
    config["namelist"]["fv_core_nml"]["external_ic"] = False
    config["namelist"]["fv_core_nml"]["nggps_ic"] = False
    config["namelist"]["fv_core_nml"]["make_nh"] = False
    config["namelist"]["fv_core_nml"]["mountain"] = True
    config["namelist"]["fv_core_nml"]["warm_start"] = True
    config["namelist"]["fv_core_nml"]["na_init"] = 0
    if MPI.COMM_WORLD.Get_rank() == 0:
        if os.path.isdir(rundir):
            shutil.rmtree(rundir)
        fv3config.write_run_directory(config, rundir)
    MPI.COMM_WORLD.barrier()
    original_path = os.getcwd()
    os.chdir(rundir)
    try:
        with redirect_stdout(os.devnull):
            fv3gfs.wrapper.initialize()
            MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.Get_rank() != 0:
            kwargs = {"verbosity": 0}
        else:
            kwargs = {"verbosity": 2}
        unittest.main(**kwargs)
    finally:
        os.chdir(original_path)
        if MPI.COMM_WORLD.Get_rank() == 0:
            shutil.rmtree(rundir)
