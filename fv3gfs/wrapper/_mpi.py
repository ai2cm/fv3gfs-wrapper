from mpi4py import MPI
import pace.util

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_tile_index(tile_rank=None, total_ranks=None):
    """Returns the tile number for a given rank and total number of ranks.

    Uses the rank for the current MPI process, and total number of processes,
    by default.
    """
    if tile_rank is None:
        tile_rank = rank
    if total_ranks is None:
        total_ranks = size
    return pace.util.get_tile_index(tile_rank, total_ranks)
