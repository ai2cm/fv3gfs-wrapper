import cftime
import xarray as xr
from typing import TextIO
from datetime import datetime
from .time import datetime64_to_datetime
from . import filesystem
from .quantity import Quantity
from ._xarray import to_dataset


# Calendar constant values copied from time_manager in FMS
THIRTY_DAY_MONTHS = 1
JULIAN = 2
GREGORIAN = 3
NOLEAP = 4
DATE_TYPES = {
    1: cftime.Datetime360Day,
    2: cftime.DatetimeJulian,
    3: cftime.DatetimeGregorian,  # Not a valid calendar in FV3GFS
    4: cftime.DatetimeNoLeap
}


def write_state(state: dict, filename: str) -> None:
    """Write a model state to a NetCDF file.
    
    Args:
        state: a model state dictionary
        filename: local or remote location to write the NetCDF file
    """
    if "time" not in state:
        raise ValueError('state must include a value for "time"')
    ds = to_dataset(state)
    with filesystem.open(filename, "wb") as f:
        ds.to_netcdf(f)


def read_state(filename: str) -> dict:
    """Read a model state from a NetCDF file.
    
    Args:
        filename: local or remote location of the NetCDF file

    Returns:
        state: a model state dictionary
    """
    out_dict = {}
    with filesystem.open(filename, "rb") as f:
        ds = xr.open_dataset(f)
    for name, value in ds.data_vars.items():
        if name == "time":
            out_dict[name] = datetime64_to_datetime(value)
        else:
            out_dict[name] = Quantity.from_data_array(value)
    return out_dict


def _get_integer_tokens(line, n_tokens):
    all_tokens = line.split()
    return [int(token) for token in all_tokens[:n_tokens]]


def get_current_date_from_coupler_res(file: TextIO) -> datetime:
    calendar_type, = _get_integer_tokens(file.readline(), 1)
    file.readline()
    year, month, day, hour, minute, second = _get_integer_tokens(file.readline(), 6)
    return DATE_TYPES[calendar_type](year, month, day, hour, minute, second)
