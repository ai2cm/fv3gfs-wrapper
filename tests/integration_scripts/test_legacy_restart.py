import logging
import sys
import fv3gfs
import os
from mpi4py import MPI
from util import fail, test_has_failed, test_data_equal


comm = MPI.COMM_WORLD


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
