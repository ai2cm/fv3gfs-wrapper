import pytest
import fv3gfs.util
import numpy as np
from fv3gfs.util._boundary_utils import _shift_boundary_slice, get_boundary_slice


def boundary_data(quantity, boundary_type, n_points, interior=True):
    boundary_slice = get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        boundary_type,
        n_points,
        interior,
    )
    return quantity.data[tuple(boundary_slice)]


@pytest.mark.cpu_only
def test_boundary_data_1_by_1_array_1_halo():
    quantity = fv3gfs.util.Quantity(
        np.random.randn(3, 3),
        dims=[fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
        units="m",
        origin=(1, 1),
        extent=(1, 1),
    )
    for side in (
        fv3gfs.util.WEST,
        fv3gfs.util.EAST,
        fv3gfs.util.NORTH,
        fv3gfs.util.SOUTH,
    ):
        assert (
            boundary_data(quantity, side, n_points=1, interior=True)
            == quantity.data[1, 1]
        )

    assert (
        boundary_data(quantity, fv3gfs.util.NORTH, n_points=1, interior=False)
        == quantity.data[2, 1]
    )
    assert (
        boundary_data(quantity, fv3gfs.util.SOUTH, n_points=1, interior=False)
        == quantity.data[0, 1]
    )
    assert (
        boundary_data(quantity, fv3gfs.util.WEST, n_points=1, interior=False)
        == quantity.data[1, 0]
    )
    assert (
        boundary_data(quantity, fv3gfs.util.EAST, n_points=1, interior=False)
        == quantity.data[1, 2]
    )


def test_boundary_data_3d_array_1_halo_z_offset_origin(numpy):
    quantity = fv3gfs.util.Quantity(
        numpy.random.randn(2, 3, 3),
        dims=[fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
        units="m",
        origin=(1, 1, 1),
        extent=(1, 1, 1),
    )
    for side in (
        fv3gfs.util.WEST,
        fv3gfs.util.EAST,
        fv3gfs.util.NORTH,
        fv3gfs.util.SOUTH,
    ):
        quantity.np.testing.assert_array_equal(
            boundary_data(quantity, side, n_points=1, interior=True),
            quantity.data[1, 1, 1],
        )

    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.NORTH, n_points=1, interior=False),
        quantity.data[1, 2, 1],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.SOUTH, n_points=1, interior=False),
        quantity.data[1, 0, 1],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.WEST, n_points=1, interior=False),
        quantity.data[1, 1, 0],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.EAST, n_points=1, interior=False),
        quantity.data[1, 1, 2],
    )


@pytest.mark.cpu_only
def test_boundary_data_2_by_2_array_2_halo():
    quantity = fv3gfs.util.Quantity(
        np.random.randn(6, 6),
        dims=[fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
        units="m",
        origin=(2, 2),
        extent=(2, 2),
    )
    for side in (
        fv3gfs.util.WEST,
        fv3gfs.util.EAST,
        fv3gfs.util.NORTH,
        fv3gfs.util.SOUTH,
    ):
        np.testing.assert_array_equal(
            boundary_data(quantity, side, n_points=2, interior=True),
            quantity.data[2:4, 2:4],
        )

    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.NORTH, n_points=1, interior=True),
        quantity.data[3:4, 2:4],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.NORTH, n_points=1, interior=False),
        quantity.data[4:5, 2:4],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.NORTH, n_points=2, interior=False),
        quantity.data[4:6, 2:4],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.SOUTH, n_points=1, interior=True),
        quantity.data[2:3, 2:4],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.SOUTH, n_points=1, interior=False),
        quantity.data[1:2, 2:4],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.SOUTH, n_points=2, interior=False),
        quantity.data[0:2, 2:4],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.WEST, n_points=2, interior=False),
        quantity.data[2:4, 0:2],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.WEST, n_points=1, interior=True),
        quantity.data[2:4, 2:3],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.WEST, n_points=1, interior=False),
        quantity.data[2:4, 1:2],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.EAST, n_points=1, interior=False),
        quantity.data[2:4, 4:5],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.EAST, n_points=2, interior=False),
        quantity.data[2:4, 4:6],
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, fv3gfs.util.EAST, n_points=1, interior=True),
        quantity.data[2:4, 3:4],
    )


@pytest.mark.parametrize(
    "dim, origin, extent, boundary_type, slice_object, reference",
    [
        pytest.param(
            fv3gfs.util.X_DIM,
            1,
            3,
            fv3gfs.util.WEST,
            slice(None, None),
            slice(1, 4),
            id="none_is_changed",
        ),
        pytest.param(
            fv3gfs.util.Y_DIM,
            1,
            3,
            fv3gfs.util.WEST,
            slice(None, None),
            slice(1, 4),
            id="perpendicular_none_is_changed",
        ),
        pytest.param(
            fv3gfs.util.X_DIM,
            1,
            3,
            fv3gfs.util.WEST,
            slice(0, 1),
            slice(1, 2),
            id="shift_to_start",
        ),
        pytest.param(
            fv3gfs.util.X_DIM,
            1,
            3,
            fv3gfs.util.WEST,
            slice(0, 2),
            slice(1, 3),
            id="shift_larger_to_start",
        ),
        pytest.param(
            fv3gfs.util.X_DIM,
            1,
            3,
            fv3gfs.util.EAST,
            slice(0, 1),
            slice(4, 5),
            id="shift_to_end",
        ),
        pytest.param(
            fv3gfs.util.X_INTERFACE_DIM,
            1,
            3,
            fv3gfs.util.WEST,
            slice(0, 1),
            slice(1, 2),
            id="shift_interface_to_start",
        ),
        pytest.param(
            fv3gfs.util.X_INTERFACE_DIM,
            1,
            3,
            fv3gfs.util.EAST,
            slice(0, 1),
            slice(4, 5),
            id="shift_interface_to_end",
        ),
        pytest.param(
            fv3gfs.util.Y_DIM,
            2,
            4,
            fv3gfs.util.SOUTH,
            slice(0, 1),
            slice(2, 3),
            id="shift_y_to_start",
        ),
        pytest.param(
            fv3gfs.util.Y_DIM,
            2,
            4,
            fv3gfs.util.NORTH,
            slice(0, 1),
            slice(6, 7),
            id="shift_y_to_end",
        ),
    ],
)
@pytest.mark.cpu_only
def test_shift_boundary_slice(
    dim, origin, extent, boundary_type, slice_object, reference
):
    result = _shift_boundary_slice(dim, origin, extent, boundary_type, slice_object)
    assert result == reference
