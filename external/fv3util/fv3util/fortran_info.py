from typing import Mapping, Iterable, Union
import yaml
import os
import copy

__all__ = [
    "properties_by_std_name",
    "get_restart_standard_names",
]

RestartProperties = Mapping[str, Mapping[str, Union[str, Iterable[str]]]]

DIRNAME = os.path.dirname(os.path.realpath(__file__))
RESTART_PROPERTIES = yaml.load(
    open(os.path.join(DIRNAME, "restart_properties.yml"), "r")
)


def get_restart_standard_names(restart_properties: RestartProperties = None):
    """Return a list of variable names needed for a smooth restart. By default uses
    restart_properties from RESTART_PROPERTIES."""
    if restart_properties is None:
        restart_properties = RESTART_PROPERTIES
    return_dict = {}
    for std_name, properties in restart_properties.items():
        return_dict[properties["restart_name"]] = std_name
    return return_dict
