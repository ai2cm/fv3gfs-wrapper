from mpi4py import MPI
import numpy as np
from fv3gfs import without_ghost_cells
import fv3gfs
import sys
from copy import deepcopy
import logging


_test_has_failed = False


def fail(message):
    global _test_has_failed
    test_has_failed = True
    logging.error(f'FAIL: {message}')


def test_has_failed():
    global _test_has_failed
    return _test_has_failed


np.random.seed(0)
required_dynamics_names = ['air_temperature', 'eastward_wind', 'northward_wind',
                           'pressure_thickness_of_atmospheric_layer']
required_physics_names = ['surface_latent_heat_flux', 'surface_sensible_heat_flux']
required_tracer_names = [
    'specific_humidity', 'cloud_water_mixing_ratio', 'rain_mixing_ratio', 'cloud_ice_mixing_ratio',
    'snow_mixing_ratio', 'graupel_mixing_ratio', 'ozone_mixing_ratio', 'cloud_amount'
]


def array_allclose(value1, value2):
    return np.all(np.isclose(value1, value2))


def test_setter(get_state, set_state):
    test_setter_all_at_once(get_state, set_state)
    test_setter_one_at_a_time(get_state, set_state)


def test_setter_all_at_once(get_state, set_state):
    original_state = get_state()
    replacement_state = deepcopy(original_state)

    for name, data_array in replacement_state.items():
        data_array.values[:] = np.random.uniform(size=data_array.shape)
    test_replacing_state(get_state, set_state, replacement_state)

    set_state(original_state)


def test_setter_one_at_a_time(get_state, set_state):
    original_state = get_state()
    replacement_state = deepcopy(original_state)

    for name, data_array in replacement_state.items():
        data_array.values[:] = np.random.uniform(size=data_array.shape)
        test_replacing_state(get_state, set_state, {name: data_array})

    set_state(original_state)


def test_replacing_state(get_state, set_state, replacement_state):
    set_state(replacement_state)
    new_state = without_ghost_cells(get_state())
    replacement_state = without_ghost_cells(replacement_state)
    for quantity_name in replacement_state.keys():
        replacement_array = replacement_state[quantity_name].values
        new_array = new_state[quantity_name].values
        if not array_allclose(new_array, replacement_array):
            fail(f'Some values for {quantity_name} are not equal to the value we tried to set (one at a time).')


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fv3gfs.initialize(comm.py2f())
    for i in range(5):
        fv3gfs.step()
    test_setter(fv3gfs.get_state, fv3gfs.set_state)
    test_setter(fv3gfs.get_dynamics_state, fv3gfs.set_state)
    test_setter(fv3gfs.get_physics_state, fv3gfs.set_state)
    test_setter(fv3gfs.get_tracer_state, fv3gfs.set_state)
    sys.exit(test_has_failed())
