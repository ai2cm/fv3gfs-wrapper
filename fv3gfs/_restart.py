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


def fix_state_dimension_names(state):
    dim_lengths = _wrapper.get_dimension_lengths()
    n_ghost = _wrapper.get_n_ghost_cells()
    dim_lengths['nx'] -= 2 * n_ghost  # restart files don't have ghost cells
    dim_lengths['ny'] -= 2 * n_ghost
    for name, data_array in state.items():
        if name != 'time':
            state[name] = fix_data_array_dimension_names(data_array, **dim_lengths)


def fix_data_array_dimension_names(data_array, nx, ny, nz, nz_soil):
    """Modify dimension names from e.g. xaxis1 to 'x' or 'x_interface' in-place.
    
    Done based on dimension length (similarly for y).

    Args:
        data_array (DataArray): the object being modified
        nx (int): the number of grid cells along the x-axis
        ny (int): the number of grid cells along the y-axis

    Returns:
        renamed_array (DataArray): new object with renamed dimensions
    """
    replacement_dict = {}
    for dim_name, length in zip(data_array.dims, data_array.shape):
        if dim_name[:5] == 'xaxis':
            if length == nx:
                replacement_dict[dim_name] = 'x'
            elif length == nx + 1:
                replacement_dict[dim_name] = 'x_interface'
            try:
                replacement_dict[dim_name] = {nx: 'x', nx + 1: 'x_interface'}[length]
            except KeyError as e:
                raise ValueError(
                    f'unable to determine dim name for dimension '
                    f'{dim_name} with length {length} (nx={nx})'
                ) from e
        elif dim_name[:5] == 'yaxis':
            try:
                replacement_dict[dim_name] = {ny: 'y', ny + 1: 'y_interface'}[length]
            except KeyError as e:
                raise ValueError(
                    f'unable to determine dim name for dimension '
                    f'{dim_name} with length {length} (ny={ny})'
                ) from e
        elif dim_name[:5] == 'zaxis':
            try:
                replacement_dict[dim_name] = {nz: 'z', nz_soil: 'z_soil'}[length]
            except KeyError as e:
                raise ValueError(
                    f'unable to determine dim name for dimension '
                    f'{dim_name} with length {length} (ny={ny})'
                ) from e
    return data_array.rename(replacement_dict)


def get_tracer_dict():
    out_dict = {}
    for name, entry in _wrapper.get_tracer_metadata().items():
        out_dict[entry['fortran_name']] = {
            'long_name': name,
            'units': entry['units'],
        }
    return out_dict


def get_restart_data(dirname, label=None):
    state_dict = {}
    for data_func in (
            get_time_data, get_fv_core_data, get_fv_srf_wind_data,
            get_fv_tracer_data, get_surface_data, get_phy_data):
        state_dict.update(data_func(dirname, label=label))
    fix_state_dimension_names(state_dict)
    return state_dict


def get_integer_tokens(line, n_tokens):
    all_tokens = line.split()
    return [int(token) for token in all_tokens[:n_tokens]]


def prepend_label(filename, label=None):
    if label is not None:
        return f'{label}.{filename}'
    else:
        return filename


def get_time_data(dirname, label=None):
    if label is None:
        filename = os.path.join(dirname, 'coupler.res')
    else:
        filename = os.path.join(dirname, label)
    return_dict = {}
    # i_to_calendar_name = {i_calendar: name for name, i_calendar in calendar_name_dict.items()}
    with open(filename, 'r') as f:
        # i_calendar = get_integer_tokens(f.readline(), 1)
        # return_dict['calendar'] = i_to_calendar_name[i_calendar]
        # year, month, day, hour, minute, second = get_integer_tokens(f.readline(), 6)
        # return_dict['start_time'] = datetime(year, month, day, hour, minute, second)
        f.readline()
        f.readline()
        year, month, day, hour, minute, second = get_integer_tokens(f.readline(), 6)
        return_dict['time'] = datetime(year, month, day, hour, minute, second)
    return return_dict


def get_fv_core_data(dirname, label=None):
    fv_core_filename = prepend_label('fv_core.res', label) + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, fv_core_filename)).isel(Time=0)
    return load_dynamics_dataset(ds)


def get_fv_srf_wind_data(dirname, label=None):
    fv_srf_wind_filename = prepend_label('fv_srf_wnd.res', label) + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, fv_srf_wind_filename)).isel(Time=0)
    return load_dynamics_dataset(ds)


def get_fv_tracer_data(dirname, label=None):
    fv_tracer_filename = prepend_label('fv_tracer.res', label) + rank_suffix
    ds = xr.open_dataset(os.path.join(dirname, fv_tracer_filename)).isel(Time=0)
    out_dict = {}

    tracer_metadata = get_tracer_dict()
    for name, data_array in ds.data_vars.items():
        long_name = tracer_metadata[name]['long_name']
        units = tracer_metadata[name]['units']
        out_dict[long_name] = data_array
        data_array.attrs['units'] = units
    return out_dict


def get_surface_data(dirname, label=None):
    sfc_data_filename = prepend_label('sfc_data', label) + rank_suffix
    if os.path.isfile(sfc_data_filename):
        ds = xr.open_dataset(os.path.join(dirname, sfc_data_filename)).isel(Time=0)
        return load_physics_dataset(ds)
    else:
        return {}  # physics data is optional


def get_phy_data(dirname, label=None):
    phy_data_filename = prepend_label('phy_data', label) + rank_suffix
    if os.path.isfile(phy_data_filename):
        ds = xr.open_dataset(os.path.join(dirname, phy_data_filename)).isel(Time=0)
        return load_physics_dataset(ds)
    else:
        return {}  # physics data is optional


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
