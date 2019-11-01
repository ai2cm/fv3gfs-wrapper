import os
from datetime import datetime
import xarray as xr
from . import _mpi as mpi
from . import _wrapper
from ._fortran_info import physics_properties, dynamics_properties

calendar_name_dict = {
    'no_calendar': 0,
    'thirty_day_months': 1,
    'julian': 2,
    'gregorian': 3,
    'noleap': 4,
}


tile = mpi.get_tile_number()
count = mpi.rank % mpi.size
if mpi.size > 6:
    rank_suffix = f'.tile{tile}.nc.{count:04}'
else:
    rank_suffix = f'.tile{tile}.nc'

coupler_filename = 'coupler_res.nc'


def get_tracer_dict():
    out_dict = {}
    for name, entry in _wrapper.get_tracer_metadata().items():
        out_dict[entry['fortran_name']] = {'long_name': name, 'units': entry['units']}
    return out_dict


def get_restart_data(dirname):
    state_dict = {}
    for data_func in (
            get_time_data, get_fv_core_data, get_fv_srf_wind_data,
            get_fv_tracer_data, get_surface_data, get_phy_data):
        state_dict.update(data_func(dirname))
    return state_dict


def get_integer_tokens(line, n_tokens):
    all_tokens = line.split()
    return [int(token) for token in all_tokens[:n_tokens]]


def get_time_data(dirname):
    filename = os.path.join(dirname, 'coupler.res')
    return_dict = {}
    i_to_calendar_name = {i_calendar: name for name, i_calendar in calendar_name_dict.items()}
    with open(filename, 'r') as f:
        # i_calendar = get_integer_tokens(f.readline(), 1)
        # return_dict['calendar'] = i_to_calendar_name[i_calendar]
        # year, month, day, hour, minute, second = get_integer_tokens(f.readline(), 6)
        # return_dict['start_time'] = datetime(year, month, day, hour, minute, second)
        f.readline()
        f.readline()
        year, month, day, hour, minute, second = get_integer_tokens(f.readline(), 6)
        return_dict['current_time'] = datetime(year, month, day, hour, minute, second)
    return return_dict


def get_fv_core_data(dirname):
    fv_core_filename = 'fv_core.res' + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, fv_core_filename))
    out_dict = {}
    out_dict['eastward_wind'] = ds['u']
    out_dict['northward_wind'] = ds['v']
    out_dict['vertical_wind'] = ds['W']
    out_dict['air_temperature'] = ds['T']
    out_dict['pressure_thickness_of_layer'] = ds['delp']
    out_dict['surface_geopotential'] = ds['phis']
    return out_dict


def get_fv_srf_wind_data(dirname):
    fv_srf_wind_filename = 'fv_srf_wnd.res' + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, fv_srf_wind_filename))
    out_dict = {}
    out_dict['surface_eastward_wind'] = ds['u_srf']
    out_dict['surface_northward_wind'] = ds['v_srf']
    return out_dict


def get_fv_tracer_data(dirname):
    fv_tracer_filename = 'fv_tracer.res' + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, fv_tracer_filename))
    out_dict = {}

    tracer_metadata = get_tracer_dict()
    for name, data_array in ds.data_vars.items():
        long_name = tracer_metadata[name]['long_name']
        units = tracer_metadata[name]['units']
        out_dict[long_name] = data_array
        data_array.attrs['units'] = units
    return out_dict


def get_surface_data(dirname):
    sfc_data_filename = 'sfc_data' + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, sfc_data_filename))
    return load_physics_dataset(ds)


def get_phy_data(dirname):
    phy_data_filename = 'phy_data' + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, phy_data_filename))
    return load_physics_dataset(ds)


def load_physics_dataset(ds):
    out_dict = {}
    remaining_names = set(ds.data_vars.keys())
    for properties in physics_properties:
        name = properties.get('restart_name', properties['fortran_name'])
        if name in ds.data_vars:
            data_array = ds.data_vars[name]
            out_dict[properties['name']] = data_array
            data_array.attrs['units'] = properties['units']
            data_array.attrs['alias'] = properties['fortran_name']
            if 'description' in properties:
                data_array.attrs['description'] = properties['description']
            remaining_names.remove(name)
    for name in remaining_names:
        out_dict[name] = ds.data_vars[name]
    return out_dict


def load_dynamics_dataset(ds):
    out_dict = {}
    remaining_names = set(ds.data_vars.keys())
    for properties in dynamics_properties:
        name = properties.get('restart_name', properties['fortran_name'])
        if name in ds.data_vars:
            data_array = ds.data_vars[name]
            out_dict[properties['name']] = data_array
            data_array.attrs['units'] = properties['units']
            data_array.attrs['alias'] = properties['fortran_name']
            if 'description' in properties:
                data_array.attrs['description'] = properties['description']
            remaining_names.remove(name)
    for name in remaining_names:
        out_dict[name] = ds.data_vars[name]
    return out_dict
