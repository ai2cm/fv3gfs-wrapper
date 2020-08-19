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

    def get_state(self, names):

        state = {
            "delp": fv3util.Quantity(self.one, units="Pa", dims=self.dims),
            "specific_humidity": fv3util.Quantity(self.one, units="", dims=self.dims),
            "cloud_water_mixing_ratio": fv3util.Quantity(
                self.one, units="", dims=self.dims
            ),
        }
        return {name: state[name] for name in names}

    def set_state(self, state):
        self.state.update(state)

    def get_tracer_metadata(self):
        return {
            "specific_humidity": {"is_water": True},
            "cloud_water_mixing_ratio": {"is_water": True},
        }


def test_set_state_mass_conserving_non_water():
    mock = MockFv3GFS()
    fv3gfs.set_state_mass_conserving(
        {"air_temperature": fv3util.Quantity(mock.one, dims=mock.dims, units="K")},
        fv3gfs=mock,
        pressure="delp",
    )

    np.testing.assert_equal(mock.state["air_temperature"].view[:], mock.one)


def test_set_state_mass_conserving_cant_set_delp():
    mock = MockFv3GFS()
    with pytest.raises(ValueError):
        fv3gfs.set_state_mass_conserving(
            {"delp": fv3util.Quantity(mock.one, dims=mock.dims, units="K")},
            fv3gfs=mock,
            pressure="delp",
        )


def test_set_state_mass_conserving_water_added():
    mock = MockFv3GFS()
    fv3gfs.set_state_mass_conserving(
        {"specific_humidity": fv3util.Quantity(2 * mock.one, dims=mock.dims, units="")},
        fv3gfs=mock,
        pressure="delp",
    )
    np.testing.assert_allclose(mock.state["delp"].view[:], 2)
