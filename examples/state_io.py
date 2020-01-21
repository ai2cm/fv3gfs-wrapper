import xarray as xr
import pickle


def dump(state, f):
    output = [{key: val.to_dict() for key, val in state_i.items()}
              for state_i in state]
    pickle.dump(output, f)


def load(f):
    output = pickle.load(f)
    return [{key: xr.DataArray.from_dict(val) for key, val in state_i.items()} for state_i in output]
