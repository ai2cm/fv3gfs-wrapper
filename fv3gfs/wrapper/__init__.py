import fv3gfs.util
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
    get_diagnostic_info,
    get_diagnostic_data,
    flags,
)
from ._restart import get_restart_names, open_restart

from .thermodynamics import set_state_mass_conserving


def get_diagnostic_by_name(
    name: str, mod_name: str = "gfs_phys"
) -> fv3gfs.util.Quantity:
    info = get_diagnostic_info()
    for idx, meta in info.items():
        if meta["mod_name"] == mod_name and meta["name"] == name:
            return get_diagnostic_data(idx)
    raise ValueError(f"There is no diagnostic {name} in module {mod_name}.")


__version__ = "0.5.0"

__all__ = list(key for key in locals().keys() if not key.startswith("_"))
