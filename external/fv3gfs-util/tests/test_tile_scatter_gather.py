import copy
import datetime
import pytest
import fv3gfs.util


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request):
    return request.param


@pytest.fixture(params=[0, 1, 3])
def n_rank_halo(request):
    return request.param


@pytest.fixture(params=[0, 3])
def n_tile_halo(request):
    return request.param


@pytest.fixture(params=["x,y", "y,x", "xi,y", "x,y,z", "z,y,x", "y,z,x"])
def dims(request):
    if request.param == "x,y":
        return [fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM]
    elif request.param == "y,x":
        return [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM]
    elif request.param == "xi,y":
        return [fv3gfs.util.X_INTERFACE_DIM, fv3gfs.util.Y_DIM]
    elif request.param == "x,y,z":
        return [fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM]
    elif request.param == "z,y,x":
        return [fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM]
    elif request.param == "y,z,x":
        return [fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM, fv3gfs.util.X_DIM]
    else:
        raise NotImplementedError()


@pytest.fixture
def units():
    return "m/s"


@pytest.fixture
def time():
    return datetime.datetime(2000, 1, 1)


@pytest.fixture()
def dim_lengths(layout):
    return {
        fv3gfs.util.X_DIM: 2 * layout[1],
        fv3gfs.util.X_INTERFACE_DIM: 2 * layout[1] + 1,
        fv3gfs.util.Y_DIM: 2 * layout[0],
        fv3gfs.util.Y_INTERFACE_DIM: 2 * layout[0] + 1,
        fv3gfs.util.Z_DIM: 3,
        fv3gfs.util.Z_INTERFACE_DIM: 4,
    }


@pytest.fixture()
def communicator_list(layout):
    total_ranks = layout[0] * layout[1]
    shared_buffer = {}
    return_list = []
    for rank in range(total_ranks):
        return_list.append(
            fv3gfs.util.TileCommunicator(
                fv3gfs.util.testing.DummyComm(rank, total_ranks, shared_buffer),
                fv3gfs.util.TilePartitioner(layout),
            )
        )
    return return_list


@pytest.fixture
def tile_extent(dims, dim_lengths):
    return_list = []
    for dim in dims:
        return_list.append(dim_lengths[dim])
    return tuple(return_list)


@pytest.fixture
def tile_state(dims, units, tile_extent):
    pass


@pytest.fixture
def scattered_states(tile_state, layout):
    pass


@pytest.fixture
def tile_quantity(dims, units, dim_lengths, tile_extent, n_tile_halo, numpy):
    return get_tile_quantity(dims, units, dim_lengths, tile_extent, n_tile_halo, numpy)


@pytest.fixture
def scattered_quantities(tile_quantity, layout, n_rank_halo, numpy):
    return_list = []
    total_ranks = layout[0] * layout[1]
    partitioner = fv3gfs.util.TilePartitioner(layout)
    for rank in range(total_ranks):
        # partitioner is tested in other tests, here we assume it works
        subtile_slice = partitioner.subtile_slice(
            rank,
            tile_dims=tile_quantity.dims,
            tile_extent=tile_quantity.extent,
            overlap=True,
        )
        subtile_view = tile_quantity.view[subtile_slice]
        subtile_quantity = get_quantity(
            tile_quantity.dims,
            tile_quantity.units,
            subtile_view.shape,
            n_rank_halo,
            numpy,
        )
        subtile_quantity.view[:] = subtile_view
        return_list.append(subtile_quantity)
    return return_list


def get_tile_quantity(dims, units, dim_lengths, tile_extent, n_halo, numpy):
    extent = [dim_lengths[dim] for dim in dims]
    quantity = get_quantity(dims, units, extent, n_halo, numpy)
    quantity.view[:] = numpy.random.randn(*quantity.extent)
    return quantity


def get_quantity(dims, units, extent, n_halo, numpy):
    shape = list(copy.deepcopy(extent))
    origin = [0 for dim in dims]
    for i, dim in enumerate(dims):
        if dim in fv3gfs.util.HORIZONTAL_DIMS:
            origin[i] += n_halo
            shape[i] += 2 * n_halo
    return fv3gfs.util.Quantity(
        numpy.zeros(shape), dims, units, origin=tuple(origin), extent=tuple(extent),
    )


def test_tile_gather_state(
    tile_quantity, scattered_quantities, communicator_list, time
):
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        state = {"time": time, "air_temperature": rank_quantity}
        out = communicator.gather_state(send_state=state)
        if communicator.rank == 0:
            result_state = out
        else:
            assert out is None
    assert result_state["time"] == time
    result = result_state["air_temperature"]
    assert result.dims == tile_quantity.dims
    assert result.units == tile_quantity.units
    assert result.extent == tile_quantity.extent
    tile_quantity.np.testing.assert_array_equal(result.view[:], tile_quantity.view[:])


def test_tile_gather_state_with_recv_state(
    tile_quantity, scattered_quantities, communicator_list, time
):
    recv_state = {"time": time, "air_temperature": copy.deepcopy(tile_quantity)}
    recv_state["air_temperature"].data[:] = -1
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        state = {"time": time, "air_temperature": rank_quantity}
        if communicator.rank == 0:
            communicator.gather_state(send_state=state, recv_state=recv_state)
        else:
            communicator.gather_state(send_state=state)
    assert recv_state["time"] == time
    result = recv_state["air_temperature"]
    assert result.dims == tile_quantity.dims
    assert result.units == tile_quantity.units
    assert result.extent == tile_quantity.extent
    tile_quantity.np.testing.assert_array_equal(result.view[:], tile_quantity.view[:])


def test_tile_gather_no_recv_quantity(
    tile_quantity, scattered_quantities, communicator_list
):
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        result = communicator.gather(send_quantity=rank_quantity)
        if communicator.rank != 0:
            assert result is None
    assert result.dims == tile_quantity.dims
    assert result.units == tile_quantity.units
    assert result.extent == tile_quantity.extent
    tile_quantity.np.testing.assert_array_equal(result.view[:], tile_quantity.view[:])


def test_tile_scatter_no_recv_quantity(
    tile_quantity, scattered_quantities, communicator_list
):
    result_list = []
    for communicator in communicator_list:
        if communicator.rank == 0:
            result_list.append(communicator.scatter(send_quantity=tile_quantity))
        else:
            result_list.append(communicator.scatter())
    for rank, (result, scattered) in enumerate(zip(result_list, scattered_quantities)):
        assert result.dims == scattered.dims
        assert result.units == scattered.units
        assert result.extent == scattered.extent
        scattered.np.testing.assert_array_equal(result.view[:], scattered.view[:])


def test_tile_scatter_with_recv_quantity(
    tile_quantity, scattered_quantities, communicator_list
):
    recv_quantities = copy.deepcopy(scattered_quantities)
    for q in recv_quantities:
        q.data[:] = 0.0
    for recv, communicator in zip(recv_quantities, communicator_list):
        if communicator.rank == 0:
            result = communicator.scatter(
                send_quantity=tile_quantity, recv_quantity=recv
            )
        else:
            result = communicator.scatter(recv_quantity=recv)
        assert result is recv
    for rank, (result, scattered) in enumerate(
        zip(recv_quantities, scattered_quantities)
    ):
        assert result.dims == scattered.dims
        assert result.units == scattered.units
        assert result.extent == scattered.extent
        scattered.np.testing.assert_array_equal(result.view[:], scattered.view[:])


def test_tile_gather_with_recv_quantity(
    tile_quantity, scattered_quantities, communicator_list
):
    recv_quantity = copy.deepcopy(tile_quantity)
    recv_quantity.data[:] = -1
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        if communicator.rank == 0:
            result = communicator.gather(
                send_quantity=rank_quantity, recv_quantity=recv_quantity
            )
        else:
            result = communicator.gather(send_quantity=rank_quantity)
            assert result is None
    assert recv_quantity.dims == tile_quantity.dims
    assert recv_quantity.units == tile_quantity.units
    assert recv_quantity.extent == tile_quantity.extent
    tile_quantity.np.testing.assert_array_equal(
        recv_quantity.view[:], tile_quantity.view[:]
    )


def test_tile_scatter_state(
    tile_quantity, scattered_quantities, communicator_list, time
):
    state = {"time": time, "air_temperature": tile_quantity}
    result_list = []
    for communicator in communicator_list:
        if communicator.rank == 0:
            result_list.append(communicator.scatter_state(send_state=state))
        else:
            result_list.append(communicator.scatter_state())
    for result_state, scattered in zip(result_list, scattered_quantities):
        assert result_state["time"] == time
        result = result_state["air_temperature"]
        assert result.dims == scattered.dims
        assert result.units == scattered.units
        assert result.extent == scattered.extent
        scattered.np.testing.assert_array_equal(result.view[:], scattered.view[:])


def test_tile_scatter_state_with_recv_state(
    tile_quantity, scattered_quantities, communicator_list, time
):
    tile_state = {"time": time, "air_temperature": tile_quantity}
    recv_quantities = copy.deepcopy(scattered_quantities)
    for q in recv_quantities:
        q.data[:] = 0.0
    for recv, communicator in zip(recv_quantities, communicator_list):
        state = {
            "time": time - datetime.timedelta(hours=1),
            "air_temperature": recv,
        }
        if communicator.rank == 0:
            result = communicator.scatter_state(send_state=tile_state, recv_state=state)
        else:
            result = communicator.scatter_state(recv_state=state)
        assert result["time"] == time
        assert result["air_temperature"] is recv
    for rank, (result, scattered) in enumerate(
        zip(recv_quantities, scattered_quantities)
    ):
        assert result.dims == scattered.dims
        assert result.units == scattered.units
        assert result.extent == scattered.extent
        scattered.np.testing.assert_array_equal(result.view[:], scattered.view[:])
