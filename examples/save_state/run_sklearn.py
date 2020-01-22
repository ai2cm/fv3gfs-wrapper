import state_io
import sys
import fsspec
import pickle
from sklearn.externals import joblib
from sklearn.utils import parallel_backend
import xarray as xr

#memory = joblib.Memory(location='cache')

SKLEARN_MODEL = "gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl"

#@memory.cache
def open_sklearn_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def rename_to_restart(state):
    return {state_io.CF_TO_RESTART_MAP.get(key, key): state[key].rename({'z': 'pfull'}) for key in state}


def rename_to_orig(state):
    return {state_io.RESTART_TO_CF_MAP.get(key, key): state[key].rename({'pfull': 'z'}) for key in state}


def predict(model, state):
    stacked = state.stack(sample=['x', 'y'])
    with parallel_backend('threading', n_jobs=1):
        output = model.predict(stacked, 'sample').unstack('sample')
    return output


def update(model, state, dt):
    renamed = rename_to_restart(state)
    state = xr.Dataset(renamed)

    tend = predict(model, state)

    updated = state.assign(sphum=state['sphum'] + tend.Q2 * dt,
                           T=state.T + tend.Q1 * dt)

    return rename_to_orig(updated), rename_to_orig(tend)
    

if __name__ == '__main__':
    with open("rundir/state.pkl", "rb") as f:
        data = state_io.load(f)

    tile = data[0]
    model = open_sklearn_model(SKLEARN_MODEL)
    preds = update(model, tile, dt=1)
    print(preds)
