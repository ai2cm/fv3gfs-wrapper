from typing import Iterable, Callable
from ..quantity import Quantity
from ._sizer import SubtileGridSizer

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


class QuantityFactory:
    def __init__(self, sizer: SubtileGridSizer, numpy):
        self._sizer = sizer
        self._numpy = numpy

    def from_backend(cls, _sizer: SubtileGridSizer, backend: str):
        numpy = StorageNumpy(backend)
        return cls(_sizer, numpy)

    def empty(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.empty, dims, units, dtype)

    def zeros(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.zeros, dims, units, dtype)

    def ones(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.ones, dims, units, dtype)

    def _allocate(
        self, allocator: Callable, dims: Iterable[str], units: str, dtype: type = float
    ):
        origin = self._sizer.get_origin(dims)
        extent = self._sizer.get_extent(dims)
        shape = self._sizer.get_shape(dims)
        return Quantity(
            allocator(shape, dtype=dtype),
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
        )
