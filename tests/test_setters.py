import unittest
import os
import sys
from copy import deepcopy
import numpy as np
import fv3gfs.wrapper
from fv3gfs.wrapper._properties import (
    DYNAMICS_PROPERTIES,
    PHYSICS_PROPERTIES,
    OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES,
)
import pace.util
from mpi4py import MPI
from util import (
    get_current_config,
    get_default_config,
    generate_data_dict,
    main,
    replace_state_with_random_values,
)


test_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PHYSICS_PROPERTIES = []
for entry in PHYSICS_PROPERTIES:
    if entry["name"] not in OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES:
        DEFAULT_PHYSICS_PROPERTIES.append(entry)


class SetterTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SetterTests, self).__init__(*args, **kwargs)
        self.tracer_data = fv3gfs.wrapper.get_tracer_metadata()
        self.dynamics_data = generate_data_dict(DYNAMICS_PROPERTIES)
        if fv3gfs.wrapper.flags.override_surface_radiative_fluxes:
            self.physics_data = generate_data_dict(PHYSICS_PROPERTIES)
        else:
            self.physics_data = generate_data_dict(DEFAULT_PHYSICS_PROPERTIES)

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

    def test_set_then_get_transposed_air_temperature(self):
        self._set_names_one_at_a_time_helper(
            ["air_temperature"],
            input_modifier=lambda quantity: pace.util.Quantity(
                np.ascontiguousarray(quantity.data.transpose()),
                dims=quantity.dims[::-1],
                units=quantity.units,
                origin=quantity.origin[::-1],
                extent=quantity.extent[::-1],
                gt4py_backend=quantity.gt4py_backend,
            ),
        )

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
            fv3gfs.wrapper.set_state(
                {
                    "non_quantity": pace.util.Quantity(
                        np.array([0.0]), dims=["dim1"], units=""
                    )
                }
            )

    def _set_names_helper(self, name_list):
        self._set_all_names_at_once_helper(name_list)
        self._set_names_one_at_a_time_helper(name_list)

    def _set_all_names_at_once_helper(self, name_list):
        replace_state = replace_state_with_random_values(name_list)
        new_state = fv3gfs.wrapper.get_state(names=name_list)
        self._check_gotten_state(new_state, name_list)
        for name, new_quantity in new_state.items():
            with self.subTest(name):
                replace_quantity = replace_state[name]
                self.assert_values_equal(new_quantity, replace_quantity)

    def _set_names_one_at_a_time_helper(self, name_list, input_modifier=None):
        old_state = fv3gfs.wrapper.get_state(names=name_list)
        self._check_gotten_state(old_state, name_list)
        for replace_name in name_list:
            with self.subTest(replace_name):
                target_quantity = deepcopy(old_state[replace_name])
                target_quantity.view[:] = np.random.uniform(size=target_quantity.extent)
                if input_modifier is not None:
                    set_quantity = input_modifier(target_quantity)
                else:
                    set_quantity = target_quantity
                replace_state = {replace_name: set_quantity}
                target_state = {replace_name: target_quantity}
                fv3gfs.wrapper.set_state(replace_state)
                new_state = fv3gfs.wrapper.get_state(names=name_list)
                self._check_gotten_state(new_state, name_list)
                for name, new_quantity in new_state.items():
                    if name == replace_name:
                        self.assertFalse(
                            new_quantity.np.all(
                                new_quantity.view[:] == old_state[name].view[:]
                            ),
                            "quantity has not changed",
                        )
                        self.assert_values_equal(new_quantity, target_state[name])
                    else:
                        self.assert_values_equal(new_quantity, old_state[name])
                old_state = new_state

    def _check_gotten_state(self, state, name_list):
        for name, value in state.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(value, pace.util.Quantity)
        for name in name_list:
            self.assertIn(name, state)
        self.assertEqual(len(name_list), len(state.keys()))

    def assert_values_equal(self, quantity1, quantity2):
        self.assertTrue(quantity1.np.all(quantity1.view[:] == quantity2.view[:]))

    def _set_unallocated_override_for_radiative_surface_flux(self, name):
        config = get_current_config()
        sizer = pace.util.SubtileGridSizer.from_namelist(config["namelist"])
        factory = pace.util.QuantityFactory(sizer, np)
        quantity = factory.zeros(["x", "y"], units="W/m**2")
        with self.assertRaisesRegex(pace.util.InvalidQuantityError, "Overriding"):
            fv3gfs.wrapper.set_state({name: quantity})

    def test_set_unallocated_override_for_radiative_surface_flux(self):
        if fv3gfs.wrapper.flags.override_surface_radiative_fluxes:
            self.skipTest("Memory is allocated for the overriding fluxes in this case.")
        for name in OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES:
            with self.subTest(name):
                self._set_unallocated_override_for_radiative_surface_flux(name)

    def test_set_surface_precipitation_rate(self):
        """Special test since this quantity is not in physics_properties.json file"""
        state = fv3gfs.wrapper.get_state(
            names=["total_precipitation", "surface_precipitation_rate"]
        )
        total_precip_old = state["total_precipitation"]
        precip_rate_new = state["surface_precipitation_rate"]
        precip_rate_new.view[:] = 2 * precip_rate_new.view[:]
        precip_rate_new_copy = deepcopy(precip_rate_new)
        fv3gfs.wrapper.set_state({"surface_precipitation_rate": precip_rate_new})
        np.testing.assert_equal(precip_rate_new.view[:], precip_rate_new_copy.view[:])
        state_new = fv3gfs.wrapper.get_state(["total_precipitation"])
        total_precip_new = state_new["total_precipitation"]
        np.testing.assert_allclose(
            2 * total_precip_old.view[:], total_precip_new.view[:]
        )


def get_override_surface_radiative_fluxes():
    """A crude way of parameterizing the setter tests for different values of
    gfs_physics_nml.override_surface_radiative_fluxes.

    See https://stackoverflow.com/questions/11380413/python-unittest-passing-arguments.
    """
    if len(sys.argv) != 2:
        raise ValueError(
            "test_setters.py requires a single argument "
            "be passed through the command line, indicating the value of "
            "the gfs_physics_nml.override_surface_radiative_fluxes flag "
            "('true' or 'false')."
        )
    override_surface_radiative_fluxes = sys.argv.pop().lower()

    # Convert string argument to bool.
    return override_surface_radiative_fluxes == "true"


if __name__ == "__main__":
    config = get_default_config()
    override_surface_radiative_fluxes = get_override_surface_radiative_fluxes()
    config["namelist"]["gfs_physics_nml"][
        "override_surface_radiative_fluxes"
    ] = override_surface_radiative_fluxes
    main(test_dir, config)
