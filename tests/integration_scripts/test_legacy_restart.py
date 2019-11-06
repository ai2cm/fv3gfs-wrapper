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


def get_timestamp(dt):
    return f'{dt.year:04}{dt.month:02}{dt.day:02}.{dt.hour:02}{dt.minute:02}{dt.second:02}'


def step_model(n_steps):
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_fortran_restart()
    comm.barrier()


if __name__ == '__main__':
    num_steps = 2
    fv3gfs.initialize()

    # Step to checkpoint
    step_model(n_steps=num_steps)
    checkpoint_data = fv3gfs.get_state(fv3gfs.get_restart_names())
    checkpoint_data_disk = fv3gfs.load_fortran_restart_folder(
        os.path.join(os.getcwd(), 'RESTART'), label=get_timestamp(checkpoint_data['time'])
    )
    test_data_equal(checkpoint_data, checkpoint_data_disk, test_case='checkpoint memory vs disk')

    # Step to end of run, continuously
    step_model(n_steps=num_steps)
    first_time_data = fv3gfs.get_state(fv3gfs.get_restart_names())

    # Return to checkpoint
    fv3gfs.set_state(checkpoint_data)
    comm.barrier()
    logging.info('comparing checkpoint to restart output after resetting to checkpoint')
    checkpoint_reset_data = fv3gfs.load_fortran_restart_folder(
        os.path.join(os.getcwd(), 'RESTART'), label=get_timestamp(first_time_data['time'])
    )
    test_data_equal(checkpoint_data, checkpoint_reset_data, test_case='checkpoint reset')

    # Step to end of run, after having restarted
    step_model(n_steps=num_steps)

    # Check continuous-run data (first time) is the same as restarted data (second time)
    second_time_data = fv3gfs.get_state(fv3gfs.get_restart_names())
    logging.info('comparing first (continuous) and second (restarted) run')
    test_data_equal(first_time_data, second_time_data, test_case='end')
    sys.exit(test_has_failed())
