"""Module for thermodynamics

This should probably be moved to fv3util once that package stabilizes
"""
import fv3gfs._wrapper
from typing import Mapping
from fv3util import Quantity


def set_state_mass_conserving(state: Mapping[str, Quantity], fv3gfs=fv3gfs._wrapper):
    """Set the state in a mass conserving way
    
    Args:
        state: a state dictionary. Any water vapor species should have the following form::

                            mass condensate or vapor
                    -------------------------------------------
                    mass vapor + mass condensate + mass dry air
            
        fv3gfs: an object implementing get_state and set_state. Defaults to
            the `fv3gfs`, but can be overrided for testing purposes.
    
    """
    pass
