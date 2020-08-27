import fv3gfs.util
import pytest


@pytest.fixture
def initial_dims(request):
    return request.param


@pytest.fixture
def initial_shape(request):
    return request.param


@pytest.fixture
def initial_data(initial_shape, numpy):
    return numpy.random.randn(*initial_shape)


@pytest.fixture
def initial_origin(request):
    return request.param


@pytest.fixture
def initial_extent(request):
    return request.param


@pytest.fixture
def transpose_order(request):
    return request.param


@pytest.fixture
def target_dims(request):
    return request.param


@pytest.fixture
def final_dims(initial_dims, transpose_order):
    return tuple(initial_dims[index] for index in transpose_order)


@pytest.fixture
def final_origin(initial_origin, transpose_order):
    return tuple(initial_origin[index] for index in transpose_order)


@pytest.fixture
def final_extent(initial_extent, transpose_order):
    return tuple(initial_extent[index] for index in transpose_order)


@pytest.fixture
def final_data(initial_data, transpose_order, numpy):
    return numpy.transpose(initial_data, transpose_order)


@pytest.fixture
def quantity(initial_data, initial_dims, initial_origin, initial_extent):
    return fv3gfs.util.Quantity(
        initial_data,
        dims=initial_dims,
        units="unit_string",
        origin=initial_origin,
        extent=initial_extent,
    )


def param_product(*param_lists):
    return_list = []
    if len(param_lists) == 0:
        return [[]]
    else:
        for item in param_lists[0]:
            for later_items in param_product(*param_lists[1:]):
                return_list.append([item] + later_items)
    if len(param_lists) == 5:
        print(return_list)
    return return_list


@pytest.mark.parametrize(
    "initial_dims, initial_shape, initial_origin, initial_extent, target_dims, transpose_order",
    [
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM),
            (6, 7),
            (1, 2),
            (2, 3),
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM),
            (0, 1),
            id="2d_keep_order"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM),
            (6, 7),
            (1, 2),
            (2, 3),
            (fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM),
            (1, 0),
            id="2d_transpose"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
            (6, 7, 8),
            (1, 2, 3),
            (2, 3, 4),
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
            (0, 1, 2),
            id="3d_keep_order"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
            (6, 7, 8),
            (1, 2, 3),
            (2, 3, 4),
            (fv3gfs.util.X_DIMS, fv3gfs.util.Y_DIMS, fv3gfs.util.Z_DIMS),
            (0, 1, 2),
            id="3d_keep_order_list_dims"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
            (6, 7, 8),
            (1, 2, 3),
            (2, 3, 4),
            (fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM),
            (2, 1, 0),
            id="3d_transpose"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
            (6, 7, 8),
            (1, 2, 3),
            (2, 3, 4),
            (fv3gfs.util.Z_DIMS, fv3gfs.util.Y_DIMS, fv3gfs.util.X_DIMS),
            (2, 1, 0),
            id="3d_transpose_list_dims"
        ),
    ],
    indirect=True
)
def test_transpose(
    quantity, target_dims, final_data, final_dims, final_origin, final_extent, numpy
):
    result = quantity.transpose(target_dims)
    numpy.testing.assert_array_equal(result.data, final_data)
    assert result.dims == final_dims
    assert result.origin == final_origin
    assert result.extent == final_extent
    assert result.units == quantity.units

@pytest.mark.parametrize(
    "initial_dims, initial_shape, initial_origin, initial_extent, target_dims, transpose_order",
    [
        pytest.param(
            (fv3gfs.util.X_DIM,),
            (6,),
            (1,),
            (2,),
            (fv3gfs.util.Y_DIM,),
            (0,),
            id="1d"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_INTERFACE_DIM),
            (6, 7),
            (1, 2),
            (2, 3),
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM),
            (0, 1),
            id="2d_switch_stagger"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM),
            (6, 7),
            (1, 2),
            (2, 3),
            (fv3gfs.util.Y_DIM, fv3gfs.util.X_INTERFACE_DIM),
            (1, 0),
            id="2d_transpose_switch_stagger"
        ),
    ],
    indirect=True
)
def test_transpose_invalid_cases(
    quantity, target_dims, final_data, final_dims, final_origin, final_extent, numpy
):
    with pytest.raises(ValueError):
        result = quantity.transpose(target_dims)
