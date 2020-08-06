from typing import Mapping, Iterable, Union
import yaml
import os

__all__ = ["RESTART_PROPERTIES"]

RestartProperties = Mapping[str, Mapping[str, Union[str, Iterable[str]]]]

DIRNAME = os.path.dirname(os.path.realpath(__file__))
RESTART_PROPERTIES = yaml.safe_load(
    open(os.path.join(DIRNAME, "restart_properties.yml"), "r")
)
