from typing import List
import copy
import fv3util.fortran_info
import fv3util
from .. import _mpi as mpi
from .. import _wrapper


def open_restart(dirname: str, partitioner: fv3util.Partitioner, label: str = '', only_names: List[str] = None) -> dict:
    """Load restart files output by the Fortran model into a state dictionary.
    
    See :py:func:`fv3gfs.set_state` if you would like to load the resulting state into
    the Fortran model.

    Args:
        dirname: location of restart files, can be local or remote
        partitioner: domain decomposition for this rank
        label: prepended string on the restart files to load
        only_names (optional): list of standard names to load

    Returns:
        state: model state dictionary
    """
    if fv3util.fortran_info.tracer_properties is None:
        properties = _metadata_to_properties(_wrapper.get_tracer_metadata())
        for entry in properties:
            entry['dims'] = ['z', 'y', 'x']
        fv3util.fortran_info.set_tracer_properties(properties)
    return fv3util.open_restart(dirname, partitioner, mpi.comm, label=label, only_names=only_names)


def _metadata_to_properties(metadata):
    return_list = []
    for name, properties in metadata.items():
        return_list.append(copy.deepcopy(properties))
        return_list[-1]['name'] = name
    return return_list
