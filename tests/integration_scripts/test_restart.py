import logging
import sys
import numpy as np
import fv3gfs
import os
from mpi4py import MPI
import xarray as xr
from util import fail, test_has_failed, fail_if_unequal

comm = MPI.COMM_WORLD


def get_restart_filename(label):
    rank = MPI.COMM_WORLD.Get_rank()
    return os.path.join(os.getcwd(), f'RESTART/{label}-rank{rank}.nc')


def save_restart(label):
    state = fv3gfs.get_state(fv3gfs.get_restart_names())
    fv3gfs.write_state(state, get_restart_filename(label))


def load_restart(label):
    return fv3gfs.read_state(get_restart_filename(label))


if __name__ == '__main__':
    num_steps = 2
    fv3gfs.initialize()
    start_time = fv3gfs.get_state(['time'])['time']
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    save_restart(label='checkpoint')
    comm.barrier()
    checkpoint_data = load_restart('checkpoint')
    if checkpoint_data['time'] <= start_time:
        fail('checkpoint time should be greater than start time')
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    save_restart('first_time')
    comm.barrier()
    first_time_data = load_restart('first_time')
    fv3gfs.set_state(checkpoint_data)
    save_restart('after_checkpoint_reset')
    comm.barrier()
    logging.info('comparing checkpoint to restart output after resetting to checkpoint')
    checkpoint_reset_data = load_restart('after_checkpoint_reset')
    fail_if_unequal(
        fv3gfs.without_ghost_cells(checkpoint_data),
        fv3gfs.without_ghost_cells(checkpoint_reset_data),
        test_case='checkpoint'
    )
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    save_restart('second_time')
    comm.barrier()
    second_time_data = load_restart('second_time')
    logging.info('comparing first (continuous) and second (restarted) run')
    fail_if_unequal(
        fv3gfs.without_ghost_cells(first_time_data),
        fv3gfs.without_ghost_cells(second_time_data),
        test_case='end'
    )
    sys.exit(test_has_failed())
