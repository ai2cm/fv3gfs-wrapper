"""Module for thermodynamics
"""
from . import _wrapper
from typing import Mapping
from pace.util import Quantity


def set_state_mass_conserving(
    state: Mapping[str, Quantity],
    fv3gfs=_wrapper,
    pressure="pressure_thickness_of_atmospheric_layer",
):
    """Set the state in a mass conserving way

    Args:
        state: a state dictionary. Any water vapor species should have the following form::

                            mass condensate or vapor
                    -------------------------------------------
                    mass vapor + mass condensate + mass dry air

        fv3gfs: an object implementing get_state, set_state, and get_tracer_metadata.
            Defaults to the Fortran wrapper, but can be overrided for testing purposes.
    """
    metadata = fv3gfs.get_tracer_metadata()
    water_variables = {k for k, v in metadata.items() if v["is_water"]}
    water_in_input = set(water_variables) & set(state)

    if pressure in state:
        raise ValueError(f"Can't set {pressure} for mass a conserving update.")
    else:
        old_state = fv3gfs.get_state([pressure, *water_variables])
        delp_old = old_state[pressure]

        # the change in total water mixing ratio is only affected by the
        # species which were modified and are therefore present in ``state``.
        total_water_old = 0.0
        for v in water_in_input:
            total_water_old += old_state[v].view[:]

        total_water_new = 0.0
        for v in water_in_input:
            total_water_new += state[v].view[:]

        delp_new = delp_old.view[:] * (1 + total_water_new - total_water_old)
        state[pressure] = Quantity(delp_new, units=delp_old.units, dims=delp_old.dims)

    fv3gfs.set_state(state)
