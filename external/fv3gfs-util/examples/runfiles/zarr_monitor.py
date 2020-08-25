import fv3gfs.util
from mpi4py import MPI

if __name__ == '__main__':
    
    monitor = fv3gfs.util.ZarrMonitor(store, partitioner, mpi_comm=MPI.COMM_WORLD)