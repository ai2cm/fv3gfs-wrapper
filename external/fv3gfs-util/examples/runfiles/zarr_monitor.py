import fv3gfs.util
from mpi4py import MPI
import numpy as np

if __name__ == '__main__':
    size = MPI.COMM_WORLD.Get_size()
    # assume square tile faces
    ranks_per_edge = int((size//6) ** 0.5)
    layout = (ranks_per_edge, ranks_per_edge)

    allocator = fv3gfs.util.QuantityFactory(sizer, np)

    partitioner = fv3gfs.util.CubedSpherePartitioner(
        fv3gfs.util.TilePartitioner(layout)
    )
    monitor = fv3gfs.util.ZarrMonitor(store, partitioner, mpi_comm=MPI.COMM_WORLD)