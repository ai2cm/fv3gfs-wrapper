import logging
import sys
import numpy as np
import fv3gfs


_test_has_failed = False


def fail(message):
    global _test_has_failed
    test_has_failed = True
    logging.error(f'FAIL: {message}')


def test_has_failed():
    global _test_has_failed
    return _test_has_failed


def test_data_equal(dict1, dict2):
    for name in dict1.keys():
        if name not in dict2.keys():
            fail(f'{name} present in first dict but not second')
    for name in dict2.keys():
        if name not in dict1.keys():
            fail(f'{name} present in second dict but not first')
    for name in dict1.keys():
        value1 = dict1[name]
        value2 = dict2[name]
        if not np.all(value1.values == value2.values):
            fail(f'{name} not equal in both datasets')


if __name__ == '__main__':
    num_steps = 2
    fv3gfs.initialize()
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_intermediate_restart()
    restart_data = fv3gfs.get_restart_data('./RESTART')
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_intermediate_restart()
    first_time_data = fv3gfs.get_restart_data('./RESTART')
    fv3gfs.set_state(restart_data)
    for i in range(num_steps):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
    fv3gfs.save_intermediate_restart()
    second_time_data = fv3gfs.get_restart_data('./RESTART')
    test_data_equal(
        fv3gfs.without_ghost_cells(first_time_data),
        fv3gfs.without_ghost_cells(second_time_data)
    )

    sys.exit(test_has_failed())
