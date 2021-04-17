from .._wrapper import get_tracer_metadata
from .._properties import (
    DYNAMICS_PROPERTIES,
    PHYSICS_PROPERTIES,
    OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES,
)

# these variables are found not to be needed for smooth restarts
# later we could represent this as a key in the dynamics/physics properties
RESTART_EXCLUDE_NAMES = [
    "convective_cloud_fraction",
    "convective_cloud_top_pressure",
    "convective_cloud_bottom_pressure",
] + OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES


def get_restart_names():
    """Return a list of variables names needed for a clean restart."""
    dynamics_names = [p["name"] for p in DYNAMICS_PROPERTIES]
    physics_names = [p["name"] for p in PHYSICS_PROPERTIES]
    tracer_names = list(get_tracer_metadata().keys())
    return_list = ["time"] + dynamics_names + tracer_names + physics_names
    for name in RESTART_EXCLUDE_NAMES:
        if name in return_list:
            return_list.remove(name)
    return return_list
