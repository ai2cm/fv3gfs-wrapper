import fv3util
import pytest
import numpy as np

# rotate_vector_data(x_data, y_data, x_metadata, y_metadata, n_clockwise_rotations)


@pytest.mark.parametrize(
    "start_data, n_clockwise_rotations, dims, target_data",
    [
        pytest.param(
            np.array([1.]),
            0,
            [fv3util.Z_DIM],
            np.array([1.]),
            id="1_value_no_rotation"
        ),
        pytest.param(
            np.array([1.]),
            1,
            [fv3util.Z_DIM],
            np.array([1.]),
            id="1_value_1_rotation"
        ),
        pytest.param(
            np.zeros([2, 3]),
            0,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.zeros([2, 3]),
            id="2d_no_rotation"
        ),
        pytest.param(
            np.zeros([2, 3]),
            1,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.zeros([3, 2]),
            id="2d_1_rotation"
        ),
        pytest.param(
            np.zeros([2, 3]),
            2,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.zeros([2, 3]),
            id="2d_2_rotations"
        ),
        pytest.param(
            np.zeros([2, 3]),
            3,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.zeros([3, 2]),
            id="2d_3_rotations"
        ),
        pytest.param(
            np.arange(5)[:, None],
            1,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.arange(5)[None, ::-1],
            id="2d_x_increasing_values"
        ),
        pytest.param(
            np.arange(5)[:, None],
            2,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.arange(5)[::-1, None],
            id="2d_x_increasing_values_double_rotate"
        ),
        pytest.param(
            np.arange(5)[None, :],
            1,
            [fv3util.X_DIM, fv3util.Y_DIM],
            np.arange(5)[:, None],
            id="2d_y_increasing_values"
        ),
    ]
)
def test_rotate_scalar_data(
        start_data, n_clockwise_rotations, dims, target_data):
    result = fv3util.communicator.rotate_scalar_data(
        start_data, dims, np, n_clockwise_rotations
    )
    np.testing.assert_array_equal(result, target_data)


@pytest.mark.parametrize(
    "start_data, n_clockwise_rotations, dims, target_data",
    [
        pytest.param(
            (np.array([1.]), np.array([1.])),
            0,
            [fv3util.Z_DIM],
            (np.array([1.]), np.array([1.])),
            id="scalar_no_rotation"
        ),
        pytest.param(
            (np.array([1.]), np.array([1.])),
            1,
            [fv3util.Z_DIM],
            (np.array([1.]), np.array([-1.])),
            id="scalar_1_rotation"
        ),
        pytest.param(
            (np.array([1.]), np.array([1.])),
            2,
            [fv3util.Z_DIM],
            (np.array([-1.]), np.array([-1.])),
            id="scalar_2_rotations"
        ),
        pytest.param(
            (np.array([1.]), np.array([1.])),
            3,
            [fv3util.Z_DIM],
            (np.array([-1.]), np.array([1.])),
            id="scalar_3_rotations"
        ),
        pytest.param(
            (np.ones([3, 2]), np.ones([2, 3])),
            3,
            [fv3util.Y_INTERFACE_DIM, fv3util.X_DIM],
            (np.ones([3, 2]) * -1, np.ones([2, 3])),
            id="2d_array_flat_values"
        ),
        pytest.param(
            (np.arange(5)[:, None], np.arange(5)[None, :]),
            1,
            [fv3util.X_DIM, fv3util.Y_DIM],
            (np.arange(5)[:, None], np.arange(5)[None, ::-1] * -1),
            id="2d_array_increasing_values"
        ),
    ]
)
def test_rotate_vector_data(
        start_data, n_clockwise_rotations, dims, target_data):
    x_data, y_data = start_data
    x_target, y_target = target_data
    x_result, y_result = fv3util.communicator.rotate_vector_data(
        x_data, y_data, n_clockwise_rotations, dims, np
    )
    np.testing.assert_array_equal(x_result, x_target)
    np.testing.assert_array_equal(y_result, y_target)

