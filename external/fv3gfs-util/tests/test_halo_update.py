import pytest
import fv3gfs.util
import copy


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request, fast):
    if fast and request.param == (1, 1):
        pytest.skip("running in fast mode")
    else:
        return request.param


@pytest.fixture
def nx_rank(n_points):
    return max(3, n_points * 2 - 1)


@pytest.fixture
def ny_rank(nx_rank):
    return nx_rank


@pytest.fixture
def nz():
    return 2


@pytest.fixture
def ny(ny_rank, layout):
    return ny_rank * layout[0]


@pytest.fixture
def nx(nx_rank, layout):
    return nx_rank * layout[1]


@pytest.fixture(params=[1, 3])
def n_points(request, fast):
    if fast and request.param == 1:
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture(params=["fewer", "more", "same"])
def n_points_update(request, n_points, fast):
    if fast and request.param == "same":
        pytest.skip("running in fast mode")
    return n_points + {"fewer": -1, "more": 1, "same": 0}[request.param]


@pytest.fixture(
    params=[
        pytest.param((fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM), id="center"),
        pytest.param(
            (fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM), id="center_3d"
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
            id="center_3d_reverse",
        ),
        pytest.param(
            (fv3gfs.util.X_DIM, fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM),
            id="center_3d_shuffle",
        ),
        pytest.param(
            (fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM), id="interface"
        ),
        pytest.param(
            (
                fv3gfs.util.Z_INTERFACE_DIM,
                fv3gfs.util.Y_INTERFACE_DIM,
                fv3gfs.util.X_INTERFACE_DIM,
            ),
            id="interface_3d",
        ),
    ]
)
def dims(request, fast):
    if fast and request.param in (
        (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM),
        (
            fv3gfs.util.Z_INTERFACE_DIM,
            fv3gfs.util.Y_INTERFACE_DIM,
            fv3gfs.util.X_INTERFACE_DIM,
        ),
    ):
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture(params=[0, 1])
def n_buffer(request):
    return request.param


@pytest.fixture
def shape(nz, ny, nx, dims, n_points, n_buffer):
    return_list = []
    length_dict = {
        fv3gfs.util.X_DIM: 2 * n_points + nx + n_buffer,
        fv3gfs.util.X_INTERFACE_DIM: 2 * n_points + nx + 1 + n_buffer,
        fv3gfs.util.Y_DIM: 2 * n_points + ny + n_buffer,
        fv3gfs.util.Y_INTERFACE_DIM: 2 * n_points + ny + 1 + n_buffer,
        fv3gfs.util.Z_DIM: nz + n_buffer,
        fv3gfs.util.Z_INTERFACE_DIM: nz + 1 + n_buffer,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def origin(n_points, dims, n_buffer):
    return_list = []
    origin_dict = {
        fv3gfs.util.X_DIM: n_points + n_buffer,
        fv3gfs.util.X_INTERFACE_DIM: n_points + n_buffer,
        fv3gfs.util.Y_DIM: n_points + n_buffer,
        fv3gfs.util.Y_INTERFACE_DIM: n_points + n_buffer,
        fv3gfs.util.Z_DIM: n_buffer,
        fv3gfs.util.Z_INTERFACE_DIM: n_buffer,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        fv3gfs.util.X_DIM: nx,
        fv3gfs.util.X_INTERFACE_DIM: nx + 1,
        fv3gfs.util.Y_DIM: ny,
        fv3gfs.util.Y_INTERFACE_DIM: ny + 1,
        fv3gfs.util.Z_DIM: nz,
        fv3gfs.util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture
def communicator_list(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3gfs.util.CubedSphereCommunicator(
                comm=fv3gfs.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
            )
        )
    return return_list


@pytest.fixture
def tile_partitioner(layout):
    return fv3gfs.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3gfs.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def updated_slice(ny, nx, dims, n_points, n_points_update):
    n_points_remain = n_points - n_points_update
    return_list = []
    length_dict = {
        fv3gfs.util.X_DIM: slice(n_points_remain, n_points + nx + n_points_update),
        fv3gfs.util.X_INTERFACE_DIM: slice(
            n_points_remain, n_points + nx + 1 + n_points_update
        ),
        fv3gfs.util.Y_DIM: slice(n_points_remain, n_points + ny + n_points_update),
        fv3gfs.util.Y_INTERFACE_DIM: slice(
            n_points_remain, n_points + ny + 1 + n_points_update
        ),
        fv3gfs.util.Z_DIM: slice(None, None),
        fv3gfs.util.Z_INTERFACE_DIM: slice(None, None),
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def remaining_ones(nz, ny, nx, n_points, n_points_update):
    width = n_points - n_points_update
    return (2 * nx + 2 * ny + 4 * width) * width


@pytest.fixture
def boundary_dict(ranks_per_tile):
    if ranks_per_tile == 1:
        return {0: fv3gfs.util.EDGE_BOUNDARY_TYPES}
    elif ranks_per_tile == 4:
        return {
            0: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHWEST, fv3gfs.util.NORTHEAST, fv3gfs.util.SOUTHEAST),
            1: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHWEST, fv3gfs.util.NORTHEAST, fv3gfs.util.SOUTHWEST),
            2: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHEAST, fv3gfs.util.SOUTHWEST, fv3gfs.util.SOUTHEAST),
            3: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHWEST, fv3gfs.util.SOUTHWEST, fv3gfs.util.SOUTHEAST),
        }
    elif ranks_per_tile == 9:
        return {
            0: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHWEST, fv3gfs.util.NORTHEAST, fv3gfs.util.SOUTHEAST),
            1: fv3gfs.util.BOUNDARY_TYPES,
            2: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHWEST, fv3gfs.util.NORTHEAST, fv3gfs.util.SOUTHWEST),
            3: fv3gfs.util.BOUNDARY_TYPES,
            4: fv3gfs.util.BOUNDARY_TYPES,
            5: fv3gfs.util.BOUNDARY_TYPES,
            6: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHEAST, fv3gfs.util.SOUTHWEST, fv3gfs.util.SOUTHEAST),
            7: fv3gfs.util.BOUNDARY_TYPES,
            8: fv3gfs.util.EDGE_BOUNDARY_TYPES
            + (fv3gfs.util.NORTHWEST, fv3gfs.util.SOUTHWEST, fv3gfs.util.SOUTHEAST),
        }
    else:
        raise NotImplementedError(ranks_per_tile)


@pytest.fixture
def depth_quantity_list(
    total_ranks, dims, units, origin, extent, shape, numpy, dtype, n_points
):
    """A list of quantities whose value indicates the distance from the computational
    domain boundary."""
    return_list = []
    for rank in range(total_ranks):
        data = numpy.empty(shape, dtype=dtype)
        data[:] = numpy.nan
        for n_inside in range(max(n_points, max(extent) // 2), -1, -1):
            for i, dim in enumerate(dims):
                if (n_inside <= extent[i] // 2) and (
                    dim in fv3gfs.util.HORIZONTAL_DIMS
                ):
                    pos = [slice(None, None)] * len(dims)
                    pos[i] = origin[i] + n_inside
                    data[tuple(pos)] = n_inside
                    pos[i] = origin[i] + extent[i] - 1 - n_inside
                    data[tuple(pos)] = n_inside
        for n_outside in range(1, n_points + 1):
            for i, dim in enumerate(dims):
                if dim in fv3gfs.util.HORIZONTAL_DIMS:
                    pos = [slice(None, None)] * len(dims)
                    pos[i] = origin[i] - n_outside
                    data[tuple(pos)] = numpy.nan
                    pos[i] = origin[i] + extent[i] + n_outside - 1
                    data[tuple(pos)] = numpy.nan
        quantity = fv3gfs.util.Quantity(
            data, dims=dims, units=units, origin=origin, extent=extent,
        )
        return_list.append(quantity)
    return return_list


@pytest.mark.parametrize(
    "layout, n_points, n_points_update, dims",
    [[(1, 1), 3, "same", [fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM]]],
    indirect=True,
)
def test_halo_update_timer(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    test that halo update produces nonzero timings for all expected labels
    """
    req_list = []
    for communicator, quantity in zip(communicator_list, zeros_quantity_list):
        req = communicator.start_halo_update(quantity, n_points_update)
        req_list.append(req)
    for req in req_list:
        req.wait()
    required_times_keys = ("pack", "unpack", "Isend", "Recv")
    for communicator in communicator_list:
        with subtests.test(rank=communicator.rank):
            assert isinstance(communicator.timer, fv3gfs.util.Timer)
            times = communicator.timer.times
            missing_keys = set(required_times_keys).difference(times.keys())
            assert len(missing_keys) == 0
            extra_keys = set(times.keys()).difference(required_times_keys)
            assert len(extra_keys) == 0
            for key in required_times_keys:
                assert times[key] > 0.0
                assert isinstance(times[key], float)


def test_depth_halo_update(
    depth_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that written values have the correct orientation"""
    sample_quantity = depth_quantity_list[0]
    y_dim, x_dim = get_horizontal_dims(sample_quantity.dims)
    y_index = sample_quantity.dims.index(y_dim)
    x_index = sample_quantity.dims.index(x_dim)
    y_extent = sample_quantity.extent[y_index]
    x_extent = sample_quantity.extent[x_index]
    req_list = []
    if 0 < n_points_update <= n_points:
        for communicator, quantity in zip(communicator_list, depth_quantity_list):
            req = communicator.start_halo_update(quantity, n_points_update)
            req_list.append(req)
        for req in req_list:
            req.wait()
        for rank, quantity in enumerate(depth_quantity_list):
            with subtests.test(rank=rank, quantity=quantity):
                for dim, extent in ((y_dim, y_extent), (x_dim, x_extent)):
                    assert numpy.all(quantity.sel(**{dim: -1}) <= 1)
                    assert numpy.all(quantity.sel(**{dim: extent}) <= 1)
                    if n_points_update >= 2:
                        assert numpy.all(quantity.sel(**{dim: -2}) <= 2)
                        assert numpy.all(quantity.sel(**{dim: extent + 1}) <= 2)
                    if n_points_update >= 3:
                        assert numpy.all(quantity.sel(**{dim: -3}) <= 3)
                        assert numpy.all(quantity.sel(**{dim: extent + 2}) <= 3)
                    if n_points_update > 3:
                        raise NotImplementedError(n_points_update)


@pytest.fixture
def zeros_quantity_list(total_ranks, dims, units, origin, extent, shape, numpy, dtype):
    """A list of quantities whose values are 0 in the computational domain and 1
    outside of it."""
    return_list = []
    for rank in range(total_ranks):
        data = numpy.ones(shape, dtype=dtype)
        quantity = fv3gfs.util.Quantity(
            data, dims=dims, units=units, origin=origin, extent=extent,
        )
        quantity.view[:] = 0.0
        return_list.append(quantity)
    return return_list


@pytest.mark.parametrize(
    "n_points, n_points_update, n_buffer", [(2, "more", 0)], indirect=True
)
def test_too_many_points_requested(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    test that an exception is raised when trying to update more halo points than exist
    """
    for communicator, quantity in zip(communicator_list, zeros_quantity_list):
        with pytest.raises(fv3gfs.util.OutOfBoundsError):
            communicator.start_halo_update(quantity, n_points_update)


def test_zeros_halo_update(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that zeros from adjacent domains get written over ones on local halo"""
    req_list = []
    if 0 < n_points_update <= n_points:
        for communicator, quantity in zip(communicator_list, zeros_quantity_list):
            req = communicator.start_halo_update(quantity, n_points_update)
            req_list.append(req)
        for req in req_list:
            req.wait()
        for rank, quantity in enumerate(zeros_quantity_list):
            boundaries = boundary_dict[rank % ranks_per_tile]
            for boundary in boundaries:
                boundary_slice = fv3gfs.util._boundary_utils.get_boundary_slice(
                    quantity.dims,
                    quantity.origin,
                    quantity.extent,
                    quantity.data.shape,
                    boundary,
                    n_points_update,
                    interior=False,
                )
                with subtests.test(
                    quantity=quantity,
                    rank=rank,
                    boundary=boundary,
                    boundary_slice=boundary_slice,
                ):
                    numpy.testing.assert_array_equal(
                        quantity.data[tuple(boundary_slice)], 0.0
                    )


def test_zeros_vector_halo_update(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that zeros from adjacent domains get written over ones on local halo"""
    x_list = zeros_quantity_list
    y_list = copy.deepcopy(x_list)
    if 0 < n_points_update <= n_points:
        req_list = []
        for communicator, y_quantity, x_quantity in zip(
            communicator_list, y_list, x_list
        ):
            req_list.append(
                communicator.start_vector_halo_update(
                    y_quantity, x_quantity, n_points_update
                )
            )
        for req in req_list:
            req.wait()
        for rank, (y_quantity, x_quantity) in enumerate(zip(y_list, x_list)):
            boundaries = boundary_dict[rank % ranks_per_tile]
            for boundary in boundaries:
                boundary_slice = fv3gfs.util._boundary_utils.get_boundary_slice(
                    x_quantity.dims,
                    x_quantity.origin,
                    x_quantity.extent,
                    x_quantity.data.shape,
                    boundary,
                    n_points_update,
                    interior=False,
                )
                with subtests.test(
                    x_quantity=x_quantity,
                    rank=rank,
                    boundary=boundary,
                    boundary_slice=boundary_slice,
                ):
                    for quantity in y_quantity, x_quantity:
                        numpy.testing.assert_array_equal(
                            quantity.data[tuple(boundary_slice)], 0.0
                        )


@pytest.mark.parametrize(
    "layout, n_points, n_points_update, dims",
    [[(1, 1), 3, "same", [fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM]]],
    indirect=True,
)
def test_vector_halo_update_timer(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    test that halo update produces nonzero timings for all expected labels
    """
    x_list = zeros_quantity_list
    y_list = copy.deepcopy(x_list)
    req_list = []
    for communicator, y_quantity, x_quantity in zip(communicator_list, y_list, x_list):
        req_list.append(
            communicator.start_vector_halo_update(
                y_quantity, x_quantity, n_points_update
            )
        )
    for req in req_list:
        req.wait()
    required_times_keys = ("pack", "unpack", "Isend", "Recv")
    for communicator in communicator_list:
        with subtests.test(rank=communicator.rank):
            assert isinstance(communicator.timer, fv3gfs.util.Timer)
            times = communicator.timer.times
            missing_keys = set(required_times_keys).difference(times.keys())
            assert len(missing_keys) == 0
            extra_keys = set(times.keys()).difference(required_times_keys)
            assert len(extra_keys) == 0
            for key in required_times_keys:
                assert times[key] > 0.0
                assert isinstance(times[key], float)


def get_horizontal_dims(dims):
    for dim in fv3gfs.util.X_DIMS:
        if dim in dims:
            x_dim = dim
            break
    else:
        raise ValueError(f"no x dimension in {dims}")
    for dim in fv3gfs.util.Y_DIMS:
        if dim in dims:
            y_dim = dim
            break
    else:
        raise ValueError(f"no y dimension in {dims}")
    return y_dim, x_dim
