from mpi4py import MPI
import fv3gfs
import xarray as xr

required_dynamics_names = ['air_temperature', 'eastward_wind', 'northward_wind', 'pressure_thickness_of_atmospheric_layer']
required_physics_names = ['surface_latent_heat_flux', 'surface_sensible_heat_flux']
required_tracer_names = [
    'specific_humidity', 'cloud_water_mixing_ratio', 'rain_mixing_ratio', 'cloud_ice_mixing_ratio',
    'snow_mixing_ratio', 'graupel_mixing_ratio', 'ozone_mixing_ratio', 'cloud_amount'
]


def test_generic_state(state):
    failure = False
    for quantity_name, data_array in state.items():
        if not isinstance(data_array, xr.DataArray):
            print(f'{quantity_name} is of type {type(data_array)} instead of DataArray')
            failure = True
        else:  # need block to check DataArray attributes
            if 'units' not in data_array.attrs:
                print(f'{quantity_name} does not have units in its attrs dictionary')
                failure = True
    return failure


def test_dynamics_state(state):
    failure = test_generic_state(state)
    for quantity_name in required_dynamics_names:
        if quantity_name not in state.keys():
            print(f'{quantity_name} not in dynamics state {dynamics_state.keys()}')
            failure = True
    return failure


def test_physics_state(state):
    failure = test_generic_state(state)
    for quantity_name in required_physics_names:
        if quantity_name not in state.keys():
            print(f'{quantity_name} not in physics state {physics_state.keys()}')
            failure = True
    return failure


if __name__ == '__main__':
    failure = False
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fv3gfs.initialize(comm.py2f())
    failure = test_generic_state(fv3gfs.get_state()) or failure
    failure = test_dynamics_state(fv3gfs.get_dynamics_state()) or failure
    failure = test_physics_state(fv3gfs.get_physics_state()) or failure
    failure = test_generic_state(fv3gfs.get_tracer_state()) or failure
    for i in range(5):
        print(f'Step {i}')
        fv3gfs.step()
    failure = test_dynamics_state(fv3gfs.get_dynamics_state()) or failure
    failure = test_generic_state(fv3gfs.get_tracer_state()) or failure
    failure = test_physics_state(fv3gfs.get_physics_state()) or failure
    failure = test_generic_state(fv3gfs.get_state()) or failure
    fv3gfs.cleanup()
    assert not failure
