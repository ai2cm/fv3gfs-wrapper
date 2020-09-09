import pytest
import fv3gfs.util


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def layout():
    return (1, 1)


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture
def tile_partitioner(layout):
    return fv3gfs.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3gfs.util.CubedSpherePartitioner(tile_partitioner)


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
def rank_quantity_list(total_ranks, numpy, dtype, units=units):
    quantity_list = []
    for rank in range(total_ranks):
        x_data = numpy.empty((3, 2), dtype=dtype)
        x_data[:] = rank
        x_quantity = fv3gfs.util.Quantity(
            x_data,
            dims=(fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_DIM),
            units=units,
            origin=(0, 0),
            extent=(3, 2),
        )
        y_data = numpy.empty((2, 3), dtype=dtype)
        y_data[:] = rank
        y_quantity = fv3gfs.util.Quantity(
            y_data,
            dims=(fv3gfs.util.Y_DIM, fv3gfs.util.X_INTERFACE_DIM),
            units=units,
            origin=(0, 0),
            extent=(2, 3),
        )
        quantity_list.append((x_quantity, y_quantity))
    return quantity_list


def rank_target_list(total_ranks, numpy):
    return_list = []
    for rank in range(total_ranks):
        if rank % 2 == 0:
            target_x = (
                numpy.array([[rank, rank], [rank, rank], [rank + 2, rank + 2]]) % 6
            )
            target_y = numpy.array([[rank, rank, rank + 1], [rank, rank, rank + 1]]) % 6
        else:
            target_x = (
                numpy.array([[rank, rank], [rank, rank], [rank + 1, rank + 1]]) % 6
            )
            target_y = numpy.array([[rank, rank, rank + 2], [rank, rank, rank + 2]]) % 6
        return_list.append((target_x, target_y))
    return return_list


@pytest.mark.filterwarnings("ignore:invalid value encountered in remainder")
def test_correct_ranks_are_synchronized_with_no_halos(
    rank_quantity_list, communicator_list, subtests, numpy, rank_target_list
):
    req_list = []
    for communicator, (x_quantity, y_quantity) in zip(
        communicator_list, rank_quantity_list
    ):
        req = communicator.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req_list.append(req)
    for req in req_list:
        req.wait()
    for (x_quantity, y_quantity), (target_x, target_y) in zip(
        rank_quantity_list, rank_target_list
    ):
        numpy.testing.assert_array_equal(x_quantity.data, target_x)
        numpy.testing.assert_array_equal(y_quantity.data, target_y)
