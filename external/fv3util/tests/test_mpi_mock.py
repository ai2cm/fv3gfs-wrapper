import tempfile
import inspect
import sys
import pytest
import numpy as np
import fv3util
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is not None and MPI.COMM_WORLD.Get_size() == 1:
    # not run as a parallel test, disable MPI tests
    MPI.Finalize()
    MPI = None


def return_worker(comm):
    return 1


def send_worker(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.


@pytest.fixture(params=['return_worker', 'send_worker'])
def worker_function():
    return globals()[request.param]


def gather_decorator(worker_function):
    def wrapped(comm):
        result = worker_function(comm)
        return comm.gather(result, root=0)
    return wrapped


@pytest.fixture
def total_ranks():
    return 6


@pytest.fixture
def dummy_list(total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(total_ranks):
        return_list.append(
            fv3util.testing.DummyComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
            )
        )
    return return_list


@pytest.fixture
def comm(worker_function, total_ranks):
    return MPI.COMM_WORLD


@pytest.fixture
def mpi_results(comm, worker_function):
    return gather_decorator(worker_function)(comm)


@pytest.fixture
def dummy_results(worker_function, dummy_list):
    result_list = []
    for comm in dummy_list:
        result_list.append(worker_function(comm))
    return result_list


@pytest.mark.skipif(
    MPI is None, reason='mpi4py is not available or pytest was not run in parallel'
)
def test_worker(comm, dummy_results, mpi_results):
    comm.barrier()
    if comm.Get_rank() == 0:
        for dummy_result, mpi_result in zip(dummy_results, mpi_results):
            if isinstance(dummy_result, np.ndarray):
                np.testing.assert_array_equal(dummy_result, mpi_result)
            else:
                assert dummy_result == mpi_result
