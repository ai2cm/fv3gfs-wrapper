import numpy as np
import pytest
import fv3util


@pytest.fixture
def quantity(request):
    return fv3util.Quantity(
        request.param[0],
        dims=request.param[1],
        units="units",
    )


# TODO: test that exception is raised if we have too many indices


@pytest.mark.parametrize(
    "quantity, view_slice, reference",
    [
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (0, 0),
            4,
            id="3_by_3_center_value"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([1, 2, 3]),
                dims=[fv3util.X_DIM],
                units="m",
                origin=(1,),
                extent=(1,),
            ),
            (-1,),
            1,
            id="3_1d_left_value"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(0, 1), slice(None, None)),
            np.array([[4]]),
            id="3_by_3_center_value_as_slice"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, 0),
            1,
            id="3_by_3_first_value"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(-1, 0), slice(0, 1)),
            np.array([[1]]),
            id="3_by_3_first_value_as_slice"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(None, None), slice(None, None)),
            np.array([[4]]),
            id="3_by_3_default_slice"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(None, None), slice(None, None)),
            np.array([[12]]),
            id="5_by_5_mostly_halo_default_slice"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(-2, 0), slice(None, None)),
            np.array([[2], [7]]),
            id="5_by_5_larger_slice"
        ),
        pytest.param(
            fv3util.Quantity(
                np.array([
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                ]),
                dims=[fv3util.X_DIM, fv3util.Y_DIM],
                units="m",
                origin=(3, 2),
                extent=(1, 1),
            ),
            (slice(-3, 0), slice(None, None)),
            np.array([[2], [7], [12]]),
            id="5_by_5_shifted_right_larger_slice"
        ),
    ]
)
def test_west(quantity, view_slice, reference):
    result = quantity.view.west[view_slice]
    quantity.np.testing.assert_array_equal(result, reference)
    # result should be a slice of the quantity memory, if it's a slice
    assert len(result.shape) == 0 or result.base is quantity.data
    transposed_quantity = fv3util.Quantity(
        quantity.data.T,
        dims=quantity.dims[::-1],
        units=quantity.units,
        origin=quantity.origin[::-1],
        extent=quantity.extent[::-1]
    )
    transposed_result = transposed_quantity.view.west[view_slice[::-1]]
    if isinstance(reference, quantity.np.ndarray):
        quantity.np.testing.assert_array_equal(transposed_result, reference.T)
    else:
        quantity.np.testing.assert_array_equal(transposed_result, reference)
