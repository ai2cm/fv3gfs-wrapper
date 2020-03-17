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


def get_mpi():
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    return MPI


def worker_setup():
    MPI = get_mpi()
    return MPI.Comm.Get_parent()


@pytest.fixture
def worker_function():
    def worker(comm):
        return 1
    return worker
    

@pytest.fixture
def parent_function(worker_function):
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as worker_script:
        worker_script.write(
            inspect.getsource(get_mpi) +
            inspect.getsource(worker_setup) +
            inspect.getsource(worker_function) +
            f"""
print('setting up')
comm = worker_setup()
print('computing result')
result = {worker_function.__name__}(comm)
print('gather')
comm.gather(result, root=0)
comm.Disconnect()
            """
        )
        comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=[worker_script.name],
            maxprocs=total_ranks
        )
        yield comm
        comm.Disconnect()


@pytest.fixture
def mpi_results(comm, parent_function):
    print('parent rank', comm.Get_rank())
    return parent_function(comm)


@pytest.fixture
def dummy_results(worker_function, dummy_list):
    result_list = []
    for comm in dummy_list:
        result_list.append(worker_function(comm))
    return result_list


@pytest.mark.skipif(MPI is None, reason='mpi4py is not available')
def test_worker(dummy_results, mpi_results):
    for dummy_result, mpi_result in zip(dummy_results, mpi_results):
        if isinstance(dummy_result, np.ndarray):
            np.testing.assert_array_equal(dummy_result, mpi_result)
        else:
            assert dummy_result == mpi_result
