import os
import json

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "dynamics_properties.json"), "r") as f:
    DYNAMICS_PROPERTIES = json.load(f)

with open(os.path.join(DIR, "physics_properties.json"), "r") as f:
    PHYSICS_PROPERTIES = json.load(f)

with open(os.path.join(DIR, "flagstruct_properties.json"), "r") as f:
    FLAGSTRUCT_PROPERTIES = json.load(f)

OVERRIDES_FOR_SURFACE_RADIATIVE_FLUXES = [
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
    "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
]

DIM_NAMES = {
    properties["name"]: properties["dims"]
    for properties in DYNAMICS_PROPERTIES + PHYSICS_PROPERTIES
}
