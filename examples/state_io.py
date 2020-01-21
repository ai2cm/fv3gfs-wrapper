import xarray as xr
import pickle


CF_TO_RESTART_MAP = {
    'specific_humidity': 'sphum', 
    'air_temperature': 'T', 
}

RESTART_TO_CF_MAP = dict(zip(CF_TO_RESTART_MAP.values(), CF_TO_RESTART_MAP.keys()))

def dump(state, f):
    output = [{key: val.to_dict() for key, val in state_i.items()}
              for state_i in state]
    pickle.dump(output, f)


def load(f):
    output = pickle.load(f)
    return [{key: xr.DataArray.from_dict(val) for key, val in state_i.items()} for state_i in output]
