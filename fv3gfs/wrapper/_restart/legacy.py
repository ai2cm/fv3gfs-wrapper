from typing import List
import fv3gfs.util
from .. import _wrapper


def open_restart(
    dirname: str,
    communicator: fv3gfs.util.CubedSphereCommunicator,
    label: str = "",
    only_names: List[str] = None,
) -> dict:
    """Load restart files output by the Fortran model into a state dictionary.
    See :py:func:`fv3gfs.set_state` if you would like to load the resulting state into
    the Fortran model.

    Args:
        dirname: location of restart files, can be local or remote
        communicator: communication object for the cubed sphere
        label: prepended string on the restart files to load
        only_names (optional): list of standard names to load

    Returns:
        state: model state dictionary
    """
    tracer_properties = _wrapper.get_tracer_metadata()
    return fv3gfs.util.open_restart(
        dirname,
        communicator,
        label=label,
        only_names=only_names,
        tracer_properties=tracer_properties,
    )
