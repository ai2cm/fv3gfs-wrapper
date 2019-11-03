import logging
import sys
import numpy as np
import fv3gfs
import os
from mpi4py import MPI
import xarray as xr


_test_has_failed = False
comm = MPI.COMM_WORLD

def fail(message):
    global _test_has_failed
    _test_has_failed = True
    logging.error(f'FAIL: {message} (rank {comm.Get_rank()})')


def test_has_failed():
    global _test_has_failed
    return _test_has_failed


def test_data_equal(dict1, dict2, test_case=''):
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


if __name__ == '__main__':
    num_steps = 2
    fv3gfs.initialize()
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_fortran_restart(label='checkpoint')
    comm.barrier()
    checkpoint_data = fv3gfs.load_fortran_restart_folder(
        os.path.join(os.getcwd(), 'RESTART'), label='checkpoint'
    )
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_fortran_restart(label='first_time')
    comm.barrier()
    first_time_data = fv3gfs.load_fortran_restart_folder(
        os.path.join(os.getcwd(), 'RESTART'), label='first_time'
    )
    fv3gfs.set_state(checkpoint_data)
    fv3gfs.save_fortran_restart(label='after_checkpoint_reset')
    comm.barrier()
    logging.info('comparing checkpoint to restart output after resetting to checkpoint')
    checkpoint_reset_data = fv3gfs.load_fortran_restart_folder(
        os.path.join(os.getcwd(), 'RESTART'), label='after_checkpoint_reset'
    )
    test_data_equal(
        fv3gfs.without_ghost_cells(checkpoint_data),
        fv3gfs.without_ghost_cells(checkpoint_reset_data),
        test_case='checkpoint'
    )
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_fortran_restart(label='second_time')
    comm.barrier()
    second_time_data = fv3gfs.load_fortran_restart_folder(
        os.path.join(os.getcwd(), 'RESTART'), label='second_time'
    )
    logging.info('comparing first (continuous) and second (restarted) run')
    test_data_equal(
        fv3gfs.without_ghost_cells(first_time_data),
        fv3gfs.without_ghost_cells(second_time_data),
        test_case='end'
    )
    sys.exit(test_has_failed())
