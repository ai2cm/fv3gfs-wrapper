from . import constants
import functools


def shift_boundary_slice_tuple(dims, origin, extent, boundary_type, slice_tuple):
    slice_list = []
    print(slice_tuple)
    for dim, entry, origin_1d, extent_1d in zip(dims, slice_tuple, origin, extent):
        slice_list.append(_shift_boundary_slice(dim, origin_1d, extent_1d, boundary_type, entry))
    print(slice_list)
    return tuple(slice_list)


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)


def _shift_boundary_slice(dim, origin, extent, boundary_type, slice_object):
    """A special case of _get_boundary_slice where one edge must be an interior or
    exterior halo point."""
    offset = _get_offset(boundary_type, dim, origin, extent)
    if isinstance(slice_object, slice):
        if slice_object.start is not None:
            print(slice_object.start, offset, origin)
            start = slice_object.start + offset
        else:
            start = slice_object.start
        if slice_object.stop is not None:
            stop = slice_object.stop + offset
        else:
            stop = slice_object.stop
        return bound_default_slice(slice(start, stop, slice_object.step), origin, origin + extent)
    else:
        return slice_object + offset  # usually an integer


def _get_offset(boundary_type, dim, origin, extent):
    boundary_at_start = boundary_at_start_of_dim(boundary_type, dim)
    if boundary_at_start is None:  # default is to index within compute domain
        return origin
    elif boundary_at_start:
        return origin
    else:
        return origin + extent


@functools.lru_cache(maxsize=None)
def get_boundary_slice(dims, origin, extent, boundary_type, n_halo, interior):
    boundary_slice = []
    for dim, origin_1d, extent_1d in zip(dims, origin, extent):
        if dim in constants.INTERFACE_DIMS:
            n_overlap = 1
        else:
            n_overlap = 0
        n_points = n_halo
        at_start = boundary_at_start_of_dim(boundary_type, dim)
        if dim not in constants.HORIZONTAL_DIMS:
            boundary_slice.append(slice(origin_1d, origin_1d + extent_1d))
        elif at_start is None:
            boundary_slice.append(slice(origin_1d, origin_1d + extent_1d))
        elif at_start:
            edge_index = origin_1d
            if interior:
                edge_index += n_overlap
                boundary_slice.append(slice(edge_index, edge_index + n_points))
            else:
                boundary_slice.append(slice(edge_index - n_points, edge_index))
        else:
            edge_index = origin_1d + extent_1d
            if interior:
                edge_index -= n_overlap
                boundary_slice.append(slice(edge_index - n_points, edge_index))
            else:
                boundary_slice.append(slice(edge_index, edge_index + n_points))
    return tuple(boundary_slice)


def boundary_at_start_of_dim(boundary: int, dim: str) -> bool:
    """
    Return True if boundary is at the start of the dimension,
    False if at the end, None if the boundary does not align with the dimension.
    """
    return BOUNDARY_AT_START_OF_DIM_MAPPING[boundary].get(dim, None)


BOUNDARY_AT_START_OF_DIM_MAPPING = {
    constants.WEST: {
        constants.X_DIM: True,
        constants.X_INTERFACE_DIM: True,
    },
    constants.EAST: {
        constants.X_DIM: False,
        constants.X_INTERFACE_DIM: False,
    },
    constants.SOUTH: {
        constants.Y_DIM: True,
        constants.Y_INTERFACE_DIM: True,
    },
    constants.NORTH: {
        constants.Y_DIM: False,
        constants.Y_INTERFACE_DIM: False,
    },
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTHWEST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.WEST]
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTHEAST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.EAST]
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTHWEST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.WEST]
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTHEAST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.EAST]
}
