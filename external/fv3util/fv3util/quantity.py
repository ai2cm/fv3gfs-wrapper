import collections
from typing import Tuple, Iterable, Dict
from types import ModuleType
import dataclasses
import numpy as np
import xarray as xr
from . import constants

try:
    import cupy
except ImportError:
    cupy = np  # avoids attribute errors while also disabling cupy support
try:
    import gt4py
except ImportError:
    gt4py = None

__all__ = ["Quantity", "QuantityMetadata"]


@dataclasses.dataclass
class QuantityMetadata:
    origin: Tuple[int, ...]
    "the start of the computational domain"
    extent: Tuple[int, ...]
    "the shape of the computational domain"
    dims: Tuple[str, ...]
    "names of each dimension"
    units: str
    "units of the quantity"
    data_type: type
    "ndarray-like type used to store the data"
    dtype: type
    "dtype of the data in the ndarray-like object"

    @property
    def dim_lengths(self) -> Dict[str, int]:
        """mapping of dimension names to their lengths"""
        return dict(zip(self.dims, self.extent))

    @property
    def np(self) -> ModuleType:
        """numpy-like module used to interact with the data"""
        if issubclass(self.data_type, np.ndarray):
            return np
        elif issubclass(self.data_type, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {self.data_type}"
            )


class ArrayView:

    def __init__(self, array):
        self._data = array

    def __getitem__(self, index):
        if not isinstance(index, collections.abc.Iterable):
            index = (index,)
        self._validate_index(index)
        return self._data[self.get_data_index(index)]

    def __setitem__(self, index, value):
        if not isinstance(index, collections.abc.Iterable):
            index = (index,)
        self._validate_index(index)
        self._data[self.get_data_index(index)] = value

    def _validate_index(self, index):
        if len(self._data.shape) == 0:
            raise NotImplementedError("ArrayView not available for 0-dimensional quantities")
        if len(index) > len(self._data.shape):
            raise IndexError(
                f"too many indices (len(index)) for array with "
                f"{len(self._data.shape)} axes"
            )
        for entry in index:
            if isinstance(entry, int):
                pass
            elif isinstance(entry, slice):
                if (entry.start is not None or entry.stop is not None):
                    raise ValueError("ArrayView only allows default slice (:), to specify extents use a tuple")
            elif isinstance(entry, tuple):
                if len(entry) != 2:
                    raise ValueError('only length-2 tuples can be used to index BoundaryArrayView')
            else:
                raise TypeError(f"received index of unexpected type {type(index)}")

    def get_data_index(self, index):
        raise NotImplementedError()


class BoundaryArrayView(ArrayView):

    def __init__(self, array, dims, origin, extent):
        super(BoundaryArrayView, self).__init__(array)
        self._dims = dims
        self._origin = origin
        self._extent = extent

    def get_data_index(self, index):
        return_index = []
        for dim, origin_1d, extent_1d, index_1d in zip(self._dims, self._origin, self._extent, index):
            return_index.append(self.get_1d_data_index(dim, origin_1d, extent_1d, index_1d))
        return tuple(return_index)

    def get_1d_data_index(self, dim, origin, extent, index):
        start_offset = self.default_start(dim, origin, extent)
        end_offset = self.default_end(dim, origin, extent)
        if isinstance(index, slice):  # only default slice is allowed
            if (index.start is not None or index.stop is not None):
                raise ValueError(
                    "ArrayView only allows default slice (:), "
                    "to specify extents use a tuple"
                )
            return slice(start_offset, end_offset)
        elif isinstance(index, tuple):
            return slice(start_offset + index[0], end_offset + index[1])
        elif isinstance(index, int):
            return start_offset + index
        else:
            raise TypeError(f"received index of unexpected type {type(index)}")

    def default_start(self, dim, origin, extent):
        return origin

    def default_end(self, dim, origin, extent):
        return origin + extent


class WestArrayView(BoundaryArrayView):

    def default_end(self, dim, origin, extent):
        if dim in (constants.X_DIM, constants.X_INTERFACE_DIM):
            return origin
        else:
            return super(WestArrayView, self).default_end(dim, origin, extent)


class EastArrayView(BoundaryArrayView):

    def default_start(self, dim, origin, extent):
        if dim in (constants.X_DIM, constants.X_INTERFACE_DIM):
            return origin + extent
        else:
            return super(EastArrayView, self).default_start(dim, origin, extent)


class NorthArrayView(BoundaryArrayView):

    def default_start(self, dim, origin, extent):
        if dim in (constants.Y_DIM, constants.Y_INTERFACE_DIM):
            return origin + extent
        else:
            return super(NorthArrayView, self).default_start(dim, origin, extent)


class SouthArrayView(BoundaryArrayView):

    def default_end(self, dim, origin, extent):
        if dim in (constants.Y_DIM, constants.Y_INTERFACE_DIM):
            return origin
        else:
            return super(SouthArrayView, self).default_start(dim, origin, extent)


# class InteriorArrayView(ArrayView):
#     def __init__(self, array, origin, extent):
#         self._data = array
#         self._origin = origin
#         self._extent = extent

#     def get_data_index(self, index):
#         if not isinstance(index, (tuple, list)):
#             index = (index,)
#         index = fill_index(index, len(self._data.shape))
#         shifted_index = []
#         for entry, origin, extent in zip(index, self._origin, self._extent):
#             if isinstance(entry, slice):
#                 shifted_slice = shift_slice(entry, origin, extent)
#                 shifted_index.append(
#                     bound_default_slice(shifted_slice, origin, origin + extent)
#                 )
#             elif entry is None:
#                 shifted_index.append(entry)
#             else:
#                 shifted_index.append(entry + origin)
#         return tuple(shifted_index)


def ensure_int_tuple(arg, arg_name):
    return_list = []
    for item in arg:
        try:
            return_list.append(int(item))
        except ValueError:
            raise TypeError(
                f"tuple arg {arg_name}={arg} contains item {item} of "
                f"unexpected type {type(item)}"
            )
    return tuple(return_list)


class LegacyBoundaryArrayView:
    def __init__(self, array, origin, extent):
        self._data = array
        self._origin = origin
        self._extent = extent

    @property
    def origin(self):
        """the start of the computational domain"""
        return self._origin

    @property
    def extent(self):
        """the shape of the computational domain"""
        return self._extent

    def __getitem__(self, index):
        if len(self.origin) == 0:
            if isinstance(index, tuple) and len(index) > 0:
                raise IndexError("more than one index given for a zero-dimension array")
            elif isinstance(index, slice) and index != slice(None, None, None):
                raise IndexError("cannot slice a zero-dimension array")
            else:
                return self._data  # array[()] does not return an ndarray
        else:
            return self._data[self._get_compute_index(index)]

    def __setitem__(self, index, value):
        self._data[self._get_compute_index(index)] = value

    def _get_compute_index(self, index):
        if not isinstance(index, (tuple, list)):
            index = (index,)
        index = fill_index(index, len(self._data.shape))
        shifted_index = []
        for entry, origin, extent in zip(index, self.origin, self.extent):
            if isinstance(entry, slice):
                shifted_slice = shift_slice(entry, origin, extent)
                shifted_index.append(
                    bound_default_slice(shifted_slice, origin, origin + extent)
                )
            elif entry is None:
                shifted_index.append(entry)
            else:
                shifted_index.append(entry + origin)
        return tuple(shifted_index)


class Quantity:
    """
    Data container for physical quantities.
    """

    def __init__(
        self,
        data,
        dims: Iterable[str],
        units: str,
        origin: Iterable[int] = None,
        extent: Iterable[int] = None,
    ):
        """
        Initialize a Quantity.

        Args:
            data: ndarray-like object containing the underlying data
            dims: dimension names for each axis
            units: units of the quantity
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
        """
        if isinstance(data, (int, float, list)):
            data = np.asarray(data)
        if origin is None:
            origin = (0,) * len(dims)  # default origin at origin of array
        else:
            origin = tuple(origin)
        if extent is None:
            extent = tuple(length - start for length, start in zip(data.shape, origin))
        else:
            extent = tuple(extent)
        self._metadata = QuantityMetadata(
            origin=ensure_int_tuple(origin, "origin"),
            extent=ensure_int_tuple(extent, "extent"),
            dims=tuple(dims),
            units=units,
            data_type=type(data),
            dtype=data.dtype,
        )
        self._attrs = {}
        self._data = data
        self._compute_domain_view = LegacyBoundaryArrayView(
            self.data, self.origin, self.extent
        )
        self._west = WestArrayView(data, dims, origin, extent)
        self._east = EastArrayView(data, dims, origin, extent)
        self._north = NorthArrayView(data, dims, origin, extent)
        self._south = SouthArrayView(data, dims, origin, extent)
        self._interior = BoundaryArrayView(data, dims, origin, extent)

    @classmethod
    def from_data_array(
        cls,
        data_array: xr.DataArray,
        origin: Iterable[int] = None,
        extent: Iterable[int] = None,
    ):
        """
        Initialize a Quantity from an xarray.DataArray.

        Args:
            data_array
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
        """
        if "units" not in data_array.attrs:
            raise ValueError("need units attribute to create Quantity from DataArray")
        return cls(
            data_array.values,
            data_array.dims,
            data_array.attrs["units"],
            origin=origin,
            extent=extent,
        )

    def __repr__(self):
        return (
            f"Quantity(\n    data=\n{self.data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )

    @property
    def west(self) -> ArrayView:
        return self._west
    
    @property
    def east(self) -> ArrayView:
        return self._east
    
    @property
    def north(self) -> ArrayView:
        return self._north
    
    @property
    def south(self) -> ArrayView:
        return self._south

    @property
    def interior(self) -> ArrayView:
        return self._interior
    
    def sel(self, **kwargs: (slice, int)) -> np.ndarray:
        """Convenience method to perform indexing on `view` using dimension names
        without knowing dimension order.
        
        Args:
            **kwargs: slice/index to retrieve for a given dimension name

        Returns:
            view_selection: an ndarray-like selection of the given indices
                on `self.view`
        """
        return self.view[tuple(kwargs.get(dim, slice(None, None)) for dim in self.dims)]

    @property
    def metadata(self) -> QuantityMetadata:
        return self._metadata

    @property
    def units(self) -> str:
        """units of the quantity"""
        return self.metadata.units

    @property
    def attrs(self) -> dict:
        return dict(**self._attrs, units=self._metadata.units)

    @property
    def dims(self) -> Tuple[str, ...]:
        """names of each dimension"""
        return self.metadata.dims

    @property
    def values(self) -> np.ndarray:
        return_array = np.asarray(self._data)
        return_array.flags.writeable = False
        return return_array

    @property
    def view(self) -> ArrayView:
        """a view into the computational domain of the underlying data"""
        return self._compute_domain_view

    @property
    def data(self) -> np.ndarray:
        """the underlying array of data"""
        return self._data

    @property
    def origin(self) -> Tuple[int, ...]:
        """the start of the computational domain"""
        return self.metadata.origin

    @property
    def extent(self) -> Tuple[int, ...]:
        """the shape of the computational domain"""
        return self.metadata.extent

    @property
    def data_array(self) -> xr.DataArray:
        return xr.DataArray(self.view[:], dims=self.dims, attrs=self.attrs)

    @property
    def np(self) -> ModuleType:
        return self.metadata.np


def fill_index(index, length):
    return tuple(index) + (slice(None, None, None),) * (length - len(index))


def shift_slice(slice_in, shift, extent):
    start = shift_index(slice_in.start, shift, extent)
    stop = shift_index(slice_in.stop, shift, extent)
    return slice(start, stop, slice_in.step)


def shift_index(current_value, shift, extent):
    if current_value is None:
        new_value = None
    else:
        new_value = current_value + shift
        if new_value < 0:
            new_value = extent + new_value
    return new_value


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)
