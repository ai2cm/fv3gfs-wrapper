from mpi4py import MPI
import numpy as np
import fv3gfs


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fv3gfs.initialize(comm.py2f())
    for i in range(fv3gfs.get_step_count()):
        fv3gfs.step()
    fv3gfs.cleanup()
