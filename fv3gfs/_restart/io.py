import xarray as xr
from datetime import datetime
import numpy as np
from .._fortran_info import dynamics_properties, physics_properties
from .._wrapper import get_tracer_metadata


def get_restart_names():
    """Return a list of variables names needed for a clean restart.
    """
    dynamics_names = [p['name'] for p in dynamics_properties]
    physics_names = [p['name'] for p in physics_properties]
    tracer_names = list(get_tracer_metadata().keys())
    return ['time'] + dynamics_names + tracer_names + physics_names


def datetime64_to_datetime(dt64):
    timestamp = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(timestamp)


def write_state(state, filename):
    if 'time' not in state:
        raise ValueError('state must include a value for "time"')
    ds = xr.Dataset(data_vars=state)
    ds.to_netcdf(filename)


def read_state(filename):
    out_dict = {}
    ds = xr.open_dataset(filename)
    for name, value in ds.data_vars.items():
        if name == 'time':
            out_dict[name] = datetime64_to_datetime(value)
        else:
            out_dict[name] = value
    return out_dict
