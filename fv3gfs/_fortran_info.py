import json
import os
import re

__all__ = ['physics_properties', 'dynamics_properties']

dirname = os.path.dirname(os.path.realpath(__file__))
physics_properties = json.load(open(os.path.join(dirname, 'physics_properties.json'), 'r'))
dynamics_properties = json.load(open(os.path.join(dirname, 'dynamics_properties.json'), 'r'))


tracer_pattern = re.compile(
    r'"TRACER",\s*"atmos_mod",\s*"(?P<fortran_name>[^"]*)"\s*"longname",\s*"(?P<long_name>[^"]*)"\s*"units",\s*"(?P<units>[^"]*)"'
)


def get_tracer_metadata():
    return_dict = {}
    with open(os.path.join(os.getcwd(), 'field_table'), 'r') as file:
        field_table = file.read()
    i_tracer = 1
    for match in tracer_pattern.finditer(field_table):
        return_dict[match.group('long_name').replace(' ', '_')] = {
            'fortran_name': match.group('fortran_name'),
            'units': match.group('units'),
            'i_tracer': i_tracer
        }
        i_tracer += 1
    return return_dict
