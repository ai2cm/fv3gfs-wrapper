import logging
import numpy as np
from mpi4py import MPI
import xarray as xr

comm = MPI.COMM_WORLD


_test_has_failed = False


def fail(message):
    global _test_has_failed
    _test_has_failed = True
    logging.error(f'FAIL: {message} (rank {comm.Get_rank()})')


def test_has_failed():
    global _test_has_failed
    return _test_has_failed


def fail_if_unequal(dict1, dict2, test_case=''):
    for name in dict1.keys():
        if name not in dict2.keys():
            fail(f'{name} present in first dict but not second')
    for name in dict2.keys():
        if name not in dict1.keys():
            fail(f'{name} present in second dict but not first')
    for name in dict1.keys():
        value1 = dict1[name]
        value2 = dict2[name]
        if name == 'time':
            if value1 != value2:
                fail(f'{test_case}: {name} not equal in both datasets, values are {value1} and {value2}')
        elif not isinstance(value1, xr.DataArray):
            fail(f'{test_case}: first value for {name} is of type {type(value1)} instead of DataArray')
        elif not isinstance(value2, xr.DataArray):
            fail(f'{test_case}: second value for {name} is of type {type(value2)} instead of DataArray')
        elif not np.all(np.isclose(value1.values, value2.values)):
            fail(f'{test_case}: {name} not close in both datasets')
        elif not np.all(value1.values == value2.values):
            fail(f'{test_case}: {name} not equal in both datasets')
        comm.barrier()
