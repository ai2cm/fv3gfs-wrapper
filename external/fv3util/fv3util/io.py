import cftime
import xarray as xr
from typing import TextIO
from .time import FMS_TO_CFTIME_TYPE
from . import filesystem
from .quantity import Quantity
from ._xarray import to_dataset


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
        ds = xr.open_dataset(f, use_cftime=True)
    for name, value in ds.data_vars.items():
        if name == "time":
            out_dict[name] = value.item()
        else:
            out_dict[name] = Quantity.from_data_array(value)
    return out_dict


def _get_integer_tokens(line, n_tokens):
    all_tokens = line.split()
    return [int(token) for token in all_tokens[:n_tokens]]


def get_current_date_from_coupler_res(file: TextIO) -> cftime.datetime:
    (fms_calendar_type,) = _get_integer_tokens(file.readline(), 1)
    file.readline()
    year, month, day, hour, minute, second = _get_integer_tokens(file.readline(), 6)
    return FMS_TO_CFTIME_TYPE[fms_calendar_type](year, month, day, hour, minute, second)
