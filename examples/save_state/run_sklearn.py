import state_io
import sys
import fsspec
import pickle
from sklearn.externals import joblib
import xarray as xr

memory = joblib.Memory(location='cache')

SKLEARN_MODEL = "gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl"

@memory.cache
def open_sklearn_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def rename_dims(x):
    return x.rename({'z': 'pfull'})


def rename_to_restart(state):
    return {state_io.CF_TO_RESTART_MAP.get(key, key): rename_dims(state[key]) for key in state}


def predict(model, state):
    state = xr.Dataset(state)
    stacked = state.stack(sample=['x', 'y'])
    output = model.predict(stacked, 'sample')
    return output.unstack('sample')


with open("rundir/state.pkl", "rb") as f:
    data = state_io.load(f)



renamed = [rename_to_restart(it) for it in data]
tile = renamed[0]
model = open_sklearn_model(SKLEARN_MODEL)
preds = predict(model, tile)
print(preds)