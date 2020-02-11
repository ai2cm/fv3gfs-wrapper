import copy
import fv3util.fortran_info
import fv3util
from .. import _mpi as mpi
from .. import _wrapper


def open_restart(dirname, label=''):
    if fv3util.fortran_info.tracer_properties is None:
        properties = _metadata_to_properties(_wrapper.get_tracer_metadata())
        for entry in properties:
            entry['dims'] = ['z', 'y', 'x']
        fv3util.fortran_info.set_tracer_properties(properties)
    return fv3util.open_restart(dirname, mpi.rank, mpi.size, label=label)


def _metadata_to_properties(metadata):
    return_list = []
    for name, properties in metadata.items():
        return_list.append(copy.deepcopy(properties))
        return_list[-1]['name'] = name
    return return_list
