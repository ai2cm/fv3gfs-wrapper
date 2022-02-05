import unittest
import os
import numpy as np
import fv3gfs.wrapper
import pace.util
from fv3gfs.wrapper._properties import (
    DYNAMICS_PROPERTIES,
    PHYSICS_PROPERTIES,
    OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES,
)
from mpi4py import MPI
from util import get_current_config, get_default_config, generate_data_dict, main


test_dir = os.path.dirname(os.path.abspath(__file__))
MM_PER_M = 1000
DEFAULT_PHYSICS_PROPERTIES = []
for entry in PHYSICS_PROPERTIES:
    if entry["name"] not in OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES:
        DEFAULT_PHYSICS_PROPERTIES.append(entry)


class GetterTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GetterTests, self).__init__(*args, **kwargs)
        self.tracer_data = fv3gfs.wrapper.get_tracer_metadata()
        self.dynamics_data = generate_data_dict(DYNAMICS_PROPERTIES)
        self.physics_data = generate_data_dict(DEFAULT_PHYSICS_PROPERTIES)
        self.mpi_comm = MPI.COMM_WORLD

    def setUp(self):
        pass

    def tearDown(self):
        self.mpi_comm.barrier()

    def test_dynamics_quantities_present_in_metadata(self):
        """Test that some small subset of dynamics names are in the data dictionary"""
        for name in ["x_wind", "y_wind", "vertical_wind", "surface_geopotential"]:
            self.assertIn(name, self.dynamics_data.keys())

    def test_physics_quantities_present_in_metadata(self):
        """Test that some small subset of physics names are in the data dictionary"""
        for name in [
            "land_sea_mask",
            "surface_temperature",
            "surface_roughness",
            "air_temperature_at_2m",
        ]:
            self.assertIn(name, self.physics_data.keys())

    def test_air_temperature_config_units(self):
        self.assertEqual(self.dynamics_data["air_temperature"]["units"], "degK")

    def test_air_temperatures_are_reasonable(self):
        """Test that air temperatures are numbers that look like air temperatures"""
        state = fv3gfs.wrapper.get_state(names=["air_temperature"])
        self.assertIn("air_temperature", state.keys())
        quantity = state["air_temperature"]
        self.assertIsInstance(quantity, pace.util.Quantity)

        self.assertEqual(quantity.units, "degK")
        self.assertTrue(quantity.np.all(quantity.view[:] > 150.0))
        self.assertTrue(quantity.np.all(quantity.view[:] < 400.0))

    def test_get_surface_geopotential(self):
        """This is a special test because it's the only 2D dynamics variable."""
        state = fv3gfs.wrapper.get_state(names=["surface_geopotential"])
        self.assertIn("surface_geopotential", state.keys())
        quantity = state["surface_geopotential"]
        self.assertIsInstance(quantity, pace.util.Quantity)

        self.assertEqual(quantity.units, "m^2 s^-2")

    def test_get_soil_temperature(self):
        """This is a special test because it uses a different vertical grid (soil levels)."""
        state = fv3gfs.wrapper.get_state(names=["soil_temperature"])
        self.assertIn("soil_temperature", state.keys())
        quantity = state["soil_temperature"]
        self.assertIsInstance(quantity, pace.util.Quantity)

        self.assertEqual(quantity.units, "degK")

    def test_get_cloud_amount(self):
        """Included because this caused a segfault at some point, as a diagnostic tracer."""
        self._get_names_helper(["cloud_amount"])

    def test_get_surface_precipitation_rate(self):
        """Special test since this quantity is not in physics_properties.json file"""
        self._get_names_helper(["surface_precipitation_rate"])
        state = fv3gfs.wrapper.get_state(
            names=["total_precipitation", "surface_precipitation_rate"]
        )
        total_precip = state["total_precipitation"]
        precip_rate = state["surface_precipitation_rate"]
        config = get_current_config()
        dt = config["namelist"]["coupler_nml"]["dt_atmos"]
        np.testing.assert_allclose(
            MM_PER_M * total_precip.view[:] / dt, precip_rate.view[:]
        )
        self.assertEqual(precip_rate.units, "mm/s")

    def test_get_hybrid_a_coordinate(self):
        self._get_names_helper(["atmosphere_hybrid_a_coordinate"])

    def test_dynamics_quantities_one_at_a_time(self):
        for name in self.dynamics_data.keys():
            self._get_names_helper([name])
            self.mpi_comm.barrier()

    def test_physics_quantities_one_at_a_time(self):
        for name in self.physics_data.keys():
            self._get_names_helper([name])
            self.mpi_comm.barrier()

    def test_tracer_quantities_one_at_a_time(self):
        for name in self.tracer_data.keys():
            self._get_names_helper([name])
            self.mpi_comm.barrier()

    def test_get_all_dynamics_quantities(self):
        self._get_names_helper(self.dynamics_data.keys())

    def test_get_all_physics_quantities(self):
        self._get_names_helper(self.physics_data.keys())

    def test_get_all_tracer_quantities(self):
        self._get_names_helper(self.tracer_data.keys())

    def test_get_restart_names(self):
        restart_names = fv3gfs.wrapper.get_restart_names()
        restart_names.remove("time")
        self._get_names_helper(restart_names)

    def test_get_all_names(self):
        self._get_names_helper(
            list(self.dynamics_data.keys())
            + list(self.physics_data.keys())
            + list(self.tracer_data.keys())
        )

    def test_get_only_some_names(self):
        all_name_list = (
            list(self.dynamics_data.keys())
            + list(self.physics_data.keys())
            + list(self.tracer_data.keys())
        )
        self._get_names_helper(all_name_list[::3])

    def _get_names_helper(self, name_list):
        state = fv3gfs.wrapper.get_state(names=name_list)
        for name, value in state.items():
            with self.subTest(name):
                self.assertIsInstance(name, str)
                self.assertIsInstance(value, pace.util.Quantity)
        for name in name_list:
            with self.subTest(name):
                self.assertIn(name, state)
        self.assertEqual(len(name_list), len(state.keys()))

    def _get_unallocated_name_helper(self, name):
        with self.assertRaisesRegex(pace.util.InvalidQuantityError, "Overriding"):
            fv3gfs.wrapper.get_state(names=[name])

    def test_unallocated_physics_properties(self):
        for name in OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES:
            with self.subTest(name):
                self._get_unallocated_name_helper(name)


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
                self.assertIsInstance(metadata["units"], str)
                self.assertIsInstance(metadata["i_tracer"], int)
                self.assertIsInstance(metadata["fortran_name"], str)

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


if __name__ == "__main__":
    config = get_default_config()
    main(test_dir, config)
