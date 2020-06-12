import fv3util
import pytest
import numpy as np


ARRAY_3_BY_3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])


@pytest.fixture
def units():
    return ""  # units don't matter for these features


@pytest.fixture
def array(request):
    return request.param


@pytest.fixture
def origin(request):
    return request.param


@pytest.fixture
def extent(request):
    return request.param


@pytest.fixture
def dims(request):
    return request.param


@pytest.fixture
def quantity(array, dims, units, origin, extent, ):
    return fv3util.Quantity(array, dims, units, origin, extent)


@pytest.mark.parametrize(
    "array, origin, extent, dims, indices, reference",
    [
        pytest.param(
            ARRAY_3_BY_3,
            (1, 1),
            (1, 1),
            [fv3util.X_DIM, fv3util.Y_DIM],
            ((-1, 0), slice(None, None)),
            ARRAY_3_BY_3[0:1, 1:2],
            id="3x3_one_value"
        ),
        pytest.param(
            ARRAY_3_BY_3,
            (1, 1),
            (1, 1),
            [fv3util.X_DIM, fv3util.Y_DIM],
            ((-0, 0), slice(None, None)),
            ARRAY_3_BY_3[1:1, 1:2],
            id="3x3_no_values"
        ),
        pytest.param(
            ARRAY_3_BY_3,
            (2, 0),
            (1, 3),
            [fv3util.X_DIM, fv3util.Y_DIM],
            ((-2, 0), slice(None, None)),
            ARRAY_3_BY_3[0:2, :],
            id="3x3_more_values"
        ),
        pytest.param(
            ARRAY_3_BY_3,
            (1, 0),
            (1, 3),
            [fv3util.X_DIM, fv3util.Y_DIM],
            ((0, 2), slice(None, None)),
            ARRAY_3_BY_3[1:3, :],
            id="3x3_interior_values"
        ),
        pytest.param(
            ARRAY_3_BY_3,
            (1, 0),
            (1, 3),
            [fv3util.X_DIM, fv3util.Y_DIM],
            ((1, 2), slice(None, None)),
            ARRAY_3_BY_3[2:3, :],
            id="3x3_far_interior_values"
        ),
    ],
    indirect=["array", "origin", "extent", "dims"]
)
def test_west(quantity, indices, reference):
    result = quantity.west[indices]
    assert result.shape == reference.shape
    assert result.dtype == reference.dtype
    np.testing.assert_array_equal(result, reference)
