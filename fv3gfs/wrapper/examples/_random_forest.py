import numpy as np
from typing import BinaryIO, TextIO, Mapping
import json
import pace.util
import os
from datetime import timedelta

try:
    import sklearn_json
except ModuleNotFoundError:
    sklearn_json = None

DIR = os.path.dirname(os.path.abspath(__file__))
RF_FILENAME = os.path.join(DIR, "rf.json")
SCALER_FILENAME = os.path.join(DIR, "scaler.npz")


def get_random_forest():
    with open(RF_FILENAME, "r") as rf_file, open(SCALER_FILENAME, "rb") as scaler_file:
        return RandomForest(rf_file, scaler_file)


class StandardScaler:
    def __init__(self, std_epsilon: np.float32 = 1e-12):
        """Standard scaler normalizer: normalizes via (x-mean)/std
        Args:
            std_epsilon: A small value that is added to the standard deviation
                of each variable to be scaled, such that no variables (even those
                that are constant across samples) are unable to be scaled due to
                having zero standard deviation. Defaults to 1e-12.
        """
        self.mean = None
        self.std = None
        self.std_epsilon: np.float32 = std_epsilon

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = data.std(axis=0).astype(np.float32) + self.std_epsilon

    def normalize(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called before normalize.")
        return (data - self.mean) / self.std

    def denormalize(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.fit must be called before denormalize.")
        return data * self.std + self.mean

    def dump(self, f: BinaryIO):
        data = {}  # type: ignore
        if self.mean is not None:
            data["mean"] = self.mean
        if self.std is not None:
            data["std"] = self.std
        return np.savez(f, **data)

    @classmethod
    def load(cls, f: BinaryIO):
        data = np.load(f)
        scaler = cls()
        scaler.mean = data.get("mean")
        scaler.std = data.get("std")
        return scaler


class RandomForest:

    inputs = ("air_temperature", "specific_humidity")

    def __init__(self, json_file: TextIO, scaler_file: BinaryIO):
        model_dict = json.load(json_file)
        self._nz = 79
        self._random_forest = sklearn_json.deserialize_model(model_dict)
        self._output_scaler = StandardScaler.load(scaler_file)

    def _pack(self, state) -> np.ndarray:
        """Turn a state into a [sample, feature] array for ML"""
        inputs = []
        for name in RandomForest.inputs:
            quantity = state[name]
            compute_domain = quantity.view[:]
            inputs.append(self._to_feature_array(compute_domain, quantity.dims))
        return np.concatenate(inputs, axis=1)

    def _unpack(self, data: np.ndarray) -> Mapping[str, np.ndarray]:
        """
        Turn a [sample, feature] array for ML into a mapping
        from state names to [sample, z] compute domain data
        """
        return {
            name: data[:, i : i + self._nz]
            for name, i in zip(RandomForest.inputs, range(0, data.shape[1], self._nz))
        }

    def _to_feature_array(self, array, dims):
        if dims.index(pace.util.Z_DIM) != 0:
            raise ValueError(
                "Example is written for data whose first dimension "
                "is the vertical, received dimensions {dims}"
            )
        array = array.transpose()  # want last dimension to be vertical
        nz = array.shape[-1]
        if nz != self._nz:
            raise ValueError(f"model is trained for 79 vertical levels, array has {nz}")

        n_samples = np.product(array.shape[:-1])
        array = array.reshape([n_samples, self._nz])
        return array

    def update(self, state, timestep: timedelta):
        input_data = self._pack(state)
        output = (
            self._output_scaler.denormalize(self._random_forest.predict(input_data))
            * timestep.total_seconds()
        )
        compute_data = self._unpack(output)
        for name, array in compute_data.items():
            # transpose to put vertical at first dimension
            state[name].view[:] += array.transpose().reshape(state[name].extent)
