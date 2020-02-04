from ._wrapper import get_n_ghost_cells
import fv3util


def without_ghost_cells(state):
    """Remove ghost cells from a state.

    Args:
        state (dict): a state dictionary with ghost cells

    Returns:
        state_without_ghost_cells (dict): a state dictionary whose DataArray objects point to
            the same underlying memory as the input state, but not the ghost cells.
    """
    n_ghost = get_n_ghost_cells()
    return fv3util.without_ghost_cells(state, n_ghost)


def with_ghost_cells(state):
    """Add ghost cells to a state.
    
    Args:
        state (dict): a state dictionary without ghost cells

    Returns:
        state_with_ghost_cells (dict): a copy of the state dictionary with ghost cells appended.
    """
    n_ghost = get_n_ghost_cells()
    return fv3util.with_ghost_cells(state, n_ghost)
