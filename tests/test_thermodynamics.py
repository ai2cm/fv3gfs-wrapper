import fv3util
import numpy as np

import fv3gfs

import pytest


class MockFv3GFS:
    def __init__(self):
        shape = [1, 1, 1]
        one = np.ones(shape, dtype=np.float32)
        dims = [fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_DIM]

        self.dims = dims
        self.one = one

    def get_state(self):

        return {
            "delp": fv3util.Quantity(self.one, units="Pa", dims=self.dims),
            "specific_humidity": fv3util.Quantity(self.one, units="", dims=self.dims),
        }

    def set_state(self, state):
        self.state = state


def test_set_state_mass_conserving_non_water():
    mock = MockFv3GFS()
    fv3gfs.set_state_mass_conserving(
        {"air_temperature": fv3util.Quantity(mock.one, dims=mock.dims, units="K")},
        fv3gfs=mock,
    )

    np.testing.assert_equal(mock.state["air_temperature"].view[:], mock.one)
