<<<<<<< HEAD
from fv3util import dynamics_properties, physics_properties
=======
import fv3util
>>>>>>> master
from .._wrapper import get_tracer_metadata

# these variables are found not to be needed for smooth restarts
# later we could represent this as a key in the dynamics/physics_PROPERTIES
RESTART_EXCLUDE_NAMES = [
    'convective_cloud_fraction',
    'convective_cloud_top_pressure',
    'convective_cloud_bottom_pressure',
]


def get_restart_names():
    """Return a list of variables names needed for a clean restart.
    """
    dynamics_names = [p['name'] for p in fv3util.DYNAMICS_PROPERTIES]
    physics_names = [p['name'] for p in fv3util.PHYSICS_PROPERTIES]
    tracer_names = list(get_tracer_metadata().keys())
    return_list = ['time'] + dynamics_names + tracer_names + physics_names
    for name in RESTART_EXCLUDE_NAMES:
        if name in return_list:
            return_list.remove(name)
    return return_list
