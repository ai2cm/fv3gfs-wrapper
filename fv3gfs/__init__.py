import sys

from ._wrapper import (
    initialize,
    step,
    step_dynamics,
    step_physics,
    save_intermediate_restart_if_enabled,
    save_fortran_restart,
    cleanup,
    get_state,
    set_state,
    get_n_ghost_cells,
    get_step_count,
    get_tracer_metadata,
    compute_physics,
    apply_physics,
)
from ._restart import get_restart_names, open_restart
from fv3util import (
    InvalidQuantityError,
    DYNAMICS_PROPERTIES,
    PHYSICS_PROPERTIES,
    read_state,
    write_state,
    apply_nudging,
    get_nudging_tendencies,
    ZarrMonitor,
    CubedSpherePartitioner,
    TilePartitioner,
    CubedSphereCommunicator,
    TileCommunicator,
    Communicator,
    Quantity,
    X_DIMS,
    Y_DIMS,
    HORIZONTAL_DIMS,
    INTERFACE_DIMS,
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
    UnitsError,
)

# capture stderr and stdout of fortran code
from ._logging import _captured_stream
for func in ["step_dynamics", "step_physics", "initialize", "cleanup"]:
    # get handle to current module
    self = sys.modules[__name__]
    setattr(self, func, _captured_stream(getattr(self, func)))

__all__ = [
    "initialize",
    "step",
    "step_dynamics",
    "step_physics",
    "save_intermediate_restart_if_enabled",
    "compute_physics",
    "apply_physics",
    "save_fortran_restart",
    "cleanup",
    "get_state",
    "set_state",
    "get_n_ghost_cells",
    "get_step_count",
    "get_tracer_metadata",
    "InvalidQuantityError",
    "DYNAMICS_PROPERTIES",
    "PHYSICS_PROPERTIES",
    "read_state",
    "write_state",
    "get_restart_names",
    "apply_nudging",
    "get_nudging_tendencies",
    "open_restart",
    "ZarrMonitor",
    "CubedSpherePartitioner",
    "TilePartitioner",
    "CubedSphereCommunicator",
    "TileCommunicator",
    "Communicator",
    "Quantity",
    "X_DIMS",
    "Y_DIMS",
    "HORIZONTAL_DIMS",
    "INTERFACE_DIMS",
    "X_DIM",
    "X_INTERFACE_DIM",
    "Y_DIM",
    "Y_INTERFACE_DIM",
    "Z_DIM",
    "Z_INTERFACE_DIM",
    "UnitsError",
]

__version__ = "0.4.1"
