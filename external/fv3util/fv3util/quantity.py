from typing import Tuple, Dict
import functools
import collections.abc
import dataclasses
import numpy as np
from . import constants
import xarray as xr
try:
    import cupy
except ImportError:
    cupy = np  # avoids attribute errors while also disabling cupy support
try:
    import gt4py
except ImportError:
    gt4py = None


class FrozenDict(collections.abc.Mapping):

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        return hash(tuple(sorted(self._d.iteritems())))


@dataclasses.dataclass
class QuantityMetadata:
    dims: Tuple[str, ...]
    dim_lengths: Dict[str, int]  # defines lengths of non-horizontal dimensions
    units: str
    data_type: type
    dtype: type

    @classmethod
    def from_quantity(cls, quantity):
        dim_lengths = dict(zip(quantity.dims, quantity.extent))
        for dim in constants.HORIZONTAL_DIMS:
            if dim in dim_lengths:
                dim_lengths.pop(dim)
        return cls(
            dims=quantity.dims,
            dim_lengths=dim_lengths,
            units=quantity.units,
            data_type=type(quantity.data),
            dtype=quantity.data.dtype
        )

    @property
    def np(self):
        if issubclass(self.data_type, np.ndarray):
            return np
        elif issubclass(self.data_type, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {self.data_type}"
            )


class Quantity:

    def __init__(self, data, dims, units, origin=None, extent=None):
        self._dims = tuple(dims)
        self._attrs = {'units': units}
        self._data = data
        if origin is None:
            self._origin = (0,) * len(dims)  # default origin at origin of array
        else:
            self._origin = tuple(origin)
        if extent is None:
            self._extent = tuple(length - start for length, start in zip(data.shape, self._origin))
        else:
            self._extent = tuple(extent)
        self._compute_domain_view = BoundedArrayView(self._data, self._origin, self._extent)

    @classmethod
    def from_data_array(cls, data_array, origin=None, extent=None):
        if 'units' not in data_array.attrs:
            raise ValueError('need units attribute to create Quantity from DataArray')
        return cls(
            data_array.values,
            data_array.dims,
            data_array.attrs['units'],
            origin=origin,
            extent=extent
        )

    def __repr__(self):
        return (
            f"Quantity(\n    data=\n{self.data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )

    def sel(self, **kwargs):
        return self.view[tuple(kwargs.get(dim, slice(None, None)) for dim in self.dims)]

    @classmethod
    def from_storage(cls, gt4py_storage, dims, units):
        raise NotImplementedError()

    @property
    def metadata(self):
        return QuantityMetadata.from_quantity(self)

    @property
    def units(self):
        return self._attrs['units']

    @property
    def attrs(self):
        return self._attrs

    @property
    def dims(self):
        return self._dims
    
    @property
    def values(self):
        return_array = np.asarray(self._data)
        return_array.flags.writeable = False
        return return_array
    
    @property
    def view(self):
        return self._compute_domain_view

    @property
    def data(self):
        return self._data

    @property
    def origin(self):
        return self._origin
    
    @property
    def extent(self):
        return self._extent
    
    @property
    def storage(self):
        raise NotImplementedError()

    @property
    def data_array(self):
        return xr.DataArray(
            self.view[:],
            dims=self.dims,
            attrs=self._attrs
        )

    @property
    def np(self):
        if isinstance(self._data, np.ndarray):
            return np
        elif isinstance(self._data, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {type(self._data)}"
            )

    def boundary_data(self, boundary_type, n_points, interior=True):
        """Return a view of points at a domain boundary.

        Args:
            boundary_type: one of 'left', 'right', 'top', 'bottom', 'top_left',
                'top_right', 'bottom_left', or 'bottom_right'
            n_points: the width of boundary to include
            interior: if True, give points inside the computational domain (default),
                otherwise give points in the halo
        """
        boundary_slice = self._get_boundary_slice(
            boundary_type, n_points, interior
        )
        return self.data[tuple(boundary_slice)]

    @functools.lru_cache(maxsize=None)
    def _get_boundary_slice(self, boundary_type, n_points, interior):
        if boundary_type in constants.EDGE_BOUNDARY_TYPES:
            dim_to_starts = DIM_TO_START_EDGE
            dim_to_ends = DIM_TO_END_EDGE
        elif boundary_type in constants.CORNER_BOUNDARY_TYPES:
            dim_to_starts = DIM_TO_START_CORNERS
            dim_to_ends = DIM_TO_END_CORNERS
        else:
            raise ValueError(
                f'invalid boundary type {boundary_type}, '
                f'must be one of {constants.BOUNDARY_TYPES}'
            )
        boundary_slice = []
        for dim, origin, extent in zip(self.dims, self.origin, self.extent):
            if dim not in constants.HORIZONTAL_DIMS:
                boundary_slice.append(slice(None, None))
            elif boundary_type in dim_to_starts[dim]:
                edge_index = self.origin[self.dims.index(dim)]
                if interior:
                    boundary_slice.append(slice(edge_index, edge_index + n_points))
                else:
                    boundary_slice.append(slice(edge_index - n_points, edge_index))
            elif boundary_type in dim_to_ends[dim]:
                edge_index = self.origin[self.dims.index(dim)] + self.extent[self.dims.index(dim)]
                if interior:
                    boundary_slice.append(slice(edge_index - n_points, edge_index))
                else:
                    boundary_slice.append(slice(edge_index, edge_index + n_points))
            else:
                boundary_slice.append(slice(origin, origin + extent))
        return boundary_slice


DIM_TO_START_EDGE = {
    constants.X_DIM: (constants.LEFT,),
    constants.X_INTERFACE_DIM: (constants.LEFT,),
    constants.Y_DIM: (constants.BOTTOM,),
    constants.Y_INTERFACE_DIM: (constants.BOTTOM,),
}

DIM_TO_END_EDGE = {
    constants.X_DIM: (constants.RIGHT,),
    constants.X_INTERFACE_DIM: (constants.RIGHT,),
    constants.Y_DIM: (constants.TOP,),
    constants.Y_INTERFACE_DIM: (constants.TOP,),
}


DIM_TO_START_CORNERS = {
    constants.X_DIM: (constants.TOP_LEFT, constants.BOTTOM_LEFT),
    constants.X_INTERFACE_DIM: (constants.TOP_LEFT, constants.BOTTOM_LEFT),
    constants.Y_DIM: (constants.BOTTOM_LEFT, constants.BOTTOM_RIGHT),
    constants.Y_INTERFACE_DIM: (constants.BOTTOM_LEFT, constants.BOTTOM_RIGHT),
}

DIM_TO_END_CORNERS = {
    constants.X_DIM: (constants.TOP_RIGHT, constants.BOTTOM_RIGHT),
    constants.X_INTERFACE_DIM: (constants.TOP_RIGHT, constants.BOTTOM_RIGHT),
    constants.Y_DIM: (constants.TOP_LEFT, constants.TOP_RIGHT),
    constants.Y_INTERFACE_DIM: (constants.TOP_LEFT, constants.TOP_RIGHT),
}


class BoundedArrayView:

    def __init__(self, array, origin, extent):
        self._data = array
        self._origin = origin
        self._extent = extent

    @property
    def origin(self):
        return self._origin
    
    @property
    def extent(self):
        return self._extent

    def __getitem__(self, index):
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
                shifted_slice = shift_slice(entry, origin)
                shifted_index.append(bound_default_slice(shifted_slice, origin, origin + extent))
            elif entry is None:
                shifted_index.append(entry)
            else:
                shifted_index.append(entry + origin)
        return tuple(shifted_index)


def fill_index(index, length):
    return tuple(index) + (slice(None, None, None),) * (length - len(index))


def shift_slice(slice_in, shift):
    if slice_in.start is None:
        start = None
    else:
        start = slice_in.start + shift
    if slice_in.stop is None:
        stop = None
    else:
        stop = slice_in.stop + shift
    return slice(start, stop, slice_in.step)


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)