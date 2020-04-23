from typing import Iterable, Callable
from ..quantity import Quantity
from ..grid import Grid

try:
    import gt4py
except ImportError:
    gt4py = None


def _wrap_storage_call(function, backend):
    def wrapped(shape, dtype=float):
        return function(backend, [0] * len(shape), shape, dtype)

    wrapped.__name__ = function.__name__
    return wrapped


class StorageNumpy:
    def __init__(self, backend):
        self.empty = _wrap_storage_call(gt4py.storage.empty, backend)
        self.zeros = _wrap_storage_call(gt4py.storage.zeros, backend)
        self.ones = _wrap_storage_call(gt4py.storage.ones, backend)


class Allocator:
    def __init__(self, grid: Grid, numpy):
        self.grid = grid
        self.numpy = numpy

    def from_backend(cls, grid: Grid, backend: str):
        numpy = StorageNumpy(backend)
        return cls(grid, numpy)

    def empty(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self.numpy.empty, dims, units, dtype)

    def zeros(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self.numpy.zeros, dims, units, dtype)

    def ones(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self.numpy.ones, dims, units, dtype)

    def _allocate(
        self, allocator: Callable, dims: Iterable[str], units: str, dtype: type = float
    ):
        origin = self.grid.get_origin(dims)
        extent = self.grid.get_extent(dims)
        shape = self.grid.get_shape(dims)
        return Quantity(
            allocator(shape, dtype=dtype),
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
        )
 