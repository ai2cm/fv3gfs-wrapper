from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_tile_number(tile_rank=None, total_ranks=None):
    """Returns the tile number for a given rank and total number of ranks.
    
    Uses the rank for the current MPI process, and total number of processes,
    by default.
    """
    if tile_rank is None:
        tile_rank = rank
    if total_ranks is None:
        total_ranks = size
    ranks_per_tile = total_ranks // 6
    if ranks_per_tile * 6 != total_ranks:
        raise ValueError('total_ranks must be evenly divisible by 6')
    return tile_rank // ranks_per_tile + 1
