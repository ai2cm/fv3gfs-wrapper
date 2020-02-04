import fv3util
from .. import _mpi as mpi


def open_restart(dirname, prefix=''):
    return fv3util.open_restart(dirname, mpi.rank, mpi.size)
