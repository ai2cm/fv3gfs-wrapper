from mpi4py import MPI
import fv3gfs
import xarray as xr
import logging
import sys


_test_has_failed = False


def fail(message):
    global _test_has_failed
    test_has_failed = True
    logging.error(f'FAIL: {message}')

def test_has_failed():
    global _test_has_failed
    return _test_has_failed


required_dynamics_names = ['air_temperature', 'eastward_wind', 'northward_wind', 'pressure_thickness_of_atmospheric_layer']
required_physics_names = ['surface_latent_heat_flux', 'surface_sensible_heat_flux']
required_tracer_names = [
    'specific_humidity', 'cloud_water_mixing_ratio', 'rain_mixing_ratio', 'cloud_ice_mixing_ratio',
    'snow_mixing_ratio', 'graupel_mixing_ratio', 'ozone_mixing_ratio', 'cloud_amount'
]


def test_generic_state(state):
    for quantity_name, data_array in state.items():
        if not isinstance(data_array, xr.DataArray):
            fail(f'{quantity_name} is of type {type(data_array)} instead of DataArray')
        else:  # need block to check DataArray attributes
            if 'units' not in data_array.attrs:
                fail(f'{quantity_name} does not have units in its attrs dictionary')


def test_dynamics_state(state):
    test_generic_state(state)
    for quantity_name in required_dynamics_names:
        if quantity_name not in state.keys():
            fail(f'{quantity_name} not in dynamics state {dynamics_state.keys()}')


def test_physics_state(state):
    test_generic_state(state)
    for quantity_name in required_physics_names:
        if quantity_name not in state.keys():
            fail(f'{quantity_name} not in physics state {physics_state.keys()}')


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fv3gfs.initialize(comm.py2f())
    test_generic_state(fv3gfs.get_state()) 
    test_dynamics_state(fv3gfs.get_dynamics_state()) 
    test_physics_state(fv3gfs.get_physics_state()) 
    test_generic_state(fv3gfs.get_tracer_state()) 
    for i in range(5):
        print(f'Step {i}')
        fv3gfs.step()
    test_dynamics_state(fv3gfs.get_dynamics_state()) 
    test_generic_state(fv3gfs.get_tracer_state()) 
    test_physics_state(fv3gfs.get_physics_state()) 
    test_generic_state(fv3gfs.get_state())
    sys.exit(test_has_failed())
