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
        value1 = dict1[name]
        value2 = dict2[name]
        if not np.all(value1.values == value2.values):
            fail(f'{name} not equal in both datasets')


if __name__ == '__main__':
    fv3gfs.initialize()
    for i in range(2):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()
    restart_data = fv3gfs.get_restart_data('./RESTART')
    for i in range(2):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()
    first_time_data = fv3gfs.get_restart_data('./RESTART')
    fv3gfs.set_state(restart_data)
    for i in range(2):
        print(f'Step {i}')
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()
    second_time_data = fv3gfs.get_restart_data('./RESTART')
    test_data_equal(first_time_data, second_time_data)

    sys.exit(test_has_failed())
