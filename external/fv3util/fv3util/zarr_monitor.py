from typing import Union, Tuple
import logging
import zarr
import numpy as np
import xarray as xr
from . import constants, utils
from .partitioner import CubedSpherePartitioner, subtile_slice

logger = logging.getLogger("fv3util")

__all__ = ["ZarrMonitor"]


class DummyComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        assert (
            root == 0
        ), "DummyComm should only be used on a single core, so root should only ever be 0"
        return value

    def barrier(self):
        return


class ZarrMonitor:
    """sympl.Monitor-style object for storing model state dictionaries in a Zarr store."""

    def __init__(
        self,
        store: Union[str, zarr.storage.MutableMapping],
        partitioner: CubedSpherePartitioner,
        mode: str = "w",
        mpi_comm=DummyComm(),
    ):
        """Create a ZarrMonitor.

        Args:
            store: Zarr store in which to store data
            partitoner: object providing grid layout information to the Monitor
            mode: mode to use to open the store. Options are as in zarr.open_group.
            mpi_comm: mpi4py comm object to use for communications. By default, will
                use a dummy comm object that works in single-core mode.
            time_chunk_size: the chunk size of the time dimension
        """
        if mpi_comm.Get_rank() == 0:
            group = zarr.open_group(store, mode=mode)
        else:
            group = None
        self._group = mpi_comm.bcast(group)
        self._comm = mpi_comm
        self._writers = None
        self.partitioner = partitioner

    def _init_writers(self, state):
        self._writers = {
            key: _ZarrVariableWriter(
                self._comm, self._group, name=key, partitioner=self.partitioner,
            )
            for key in set(state.keys()).difference(["time"])
        }
        self._writers["time"] = _ZarrTimeWriter(
            self._comm, self._group, name="time", partitioner=self.partitioner,
        )

    def _check_writers(self, state):
        extra_names = set(state.keys()).difference(self._writers.keys())
        if len(extra_names) != 0:
            raise ValueError(
                f"provided state has keys {extra_names} "
                "that were not present in earlier states"
            )
        missing_names = set(self._writers.keys()).difference(state.keys())
        if len(missing_names) != 0:
            raise ValueError(
                f"provided state is missing keys {missing_names} "
                "that were present in earlier states"
            )

    def _ensure_writers_are_consistent(self, state):
        if self._writers is None:
            self._init_writers(state)
        else:
            self._check_writers(state)

    def store(self, state: dict) -> None:
        """Append the model state dictionary to the zarr store.

        Requires the state contain the same quantities with the same metadata as the
        first time this is called. Quantities are stored with dimensions [time, rank]
        followed by the dimensions included in any one state snapshot. The one exception
        is "time" which is stored with dimensions [time].
        """
        self._ensure_writers_are_consistent(state)
        for name, quantity in state.items():
            self._writers[name].append(quantity)


class _ZarrVariableWriter:
    def __init__(self, comm, group, name, partitioner):
        self.i_time = 0
        self.comm = comm
        self.group = group
        self.name = name
        self.array = None

        self._prepend_shape = (1, 6)
        self._prepend_chunks = (1, 1)
        self._y_chunks = partitioner.tile.layout[0]
        self._x_chunks = partitioner.tile.layout[1]
        self._PREPEND_DIMS = ("time", "tile")
        self._partitioner = partitioner

    @property
    def partitioner(self):
        return self._partitioner

    @property
    def rank(self):
        return self.comm.Get_rank()

    def _init_zarr(self, quantity):
        if self.rank == 0:
            self._init_zarr_root(quantity)
            self.array.attrs.update(self._get_attrs(quantity))
        self.sync_array()

    def _init_zarr_root(self, quantity):
        tile_shape = self._partitioner.tile.tile_extent(quantity.metadata)
        chunks = self._prepend_chunks + array_chunks(
            self._partitioner.layout, tile_shape, quantity.dims
        )
        self.array = self.group.create_dataset(
            self.name,
            shape=self._prepend_shape + tile_shape,
            dtype=quantity.data.dtype,
            chunks=chunks,
            fill_value=None,
        )

    def sync_array(self):
        self.array = self.comm.bcast(self.array, root=0)

    def append(self, quantity):
        # can't just use zarr_array.append because we only want to
        # extend the dimension once, from the master rank
        if self.array is None:
            self._init_zarr(quantity)

        if self.i_time >= self.array.shape[0] and self.rank == 0:
            new_shape = list(
                self._prepend_shape
                + self._partitioner.tile.tile_extent(quantity.metadata)
            )
            new_shape[0] = self.i_time + 1
            self.array.resize(*new_shape)
            self._ensure_compatible_attrs(quantity)
        self.sync_array()

        target_slice = (
            self.i_time,
            self._partitioner.tile_index(self.rank),
        ) + subtile_slice(
            quantity.dims,
            self.array.shape[2:],  # remove time and tile dimensions
            self.partitioner.layout,
            self.partitioner.tile.subtile_index(self.rank),
            overlap=False,
        )

        from_slice = _get_from_slice(target_slice)
        logger.debug(
            f"assigning data from subtile slice {from_slice} to target slice {target_slice}"
        )
        self.array[target_slice] = np.asarray(quantity.view[:])[from_slice]
        self.i_time += 1

    def _get_attrs(self, quantity):
        return {
            "_ARRAY_DIMENSIONS": list(self._PREPEND_DIMS + quantity.dims),
            **quantity.attrs,
        }

    def _ensure_compatible_attrs(self, new_quantity):
        new_attrs = self._get_attrs(new_quantity)
        if dict(self.array.attrs) != new_attrs:
            raise ValueError(
                f"value for {self.name} with attrs {new_attrs} "
                f"does not match previously stored attrs {dict(self.array.attrs)}"
            )


def array_chunks(
    layout: Tuple[int, int],
    tile_array_shape: Tuple[int, ...],
    array_dims: Tuple[str, ...],
):
    layout_by_dims = utils.list_by_dims(array_dims, layout, 1)
    chunks_list = []
    for extent, dim, n_ranks in zip(tile_array_shape, array_dims, layout_by_dims):
        if dim in constants.INTERFACE_DIMS:
            chunks_list.append(int((extent - 1) // n_ranks))
        else:
            chunks_list.append(int(extent // n_ranks))
    return tuple(chunks_list)


def _get_from_slice(target_slice):
    return_list = []
    for entry in target_slice:
        if isinstance(entry, slice):
            return_list.append(slice(0, entry.stop - entry.start))
    return tuple(return_list)


class _ZarrTimeWriter(_ZarrVariableWriter):

    _TIME_CHUNK_SIZE = 1024

    def __init__(self, *args, **kwargs):
        super(_ZarrTimeWriter, self).__init__(*args, **kwargs)
        self._prepend_shape = (1,)
        self._prepend_chunks = (self._TIME_CHUNK_SIZE,)
        self._PREPEND_DIMS = ("time",)

    def _init_zarr_root(self, array):
        shape = self._prepend_shape
        chunks = self._prepend_chunks
        self.array = self.group.create_dataset(
            self.name, shape=shape, dtype=array.dtype, chunks=chunks
        )

    def append(self, time):
        array = xr.DataArray(np.datetime64(time))
        if self.array is None:
            self._init_zarr(array)
        if self.i_time >= self.array.shape[0] and self.rank == 0:
            new_shape = (self.i_time + 1,)
            self.array.resize(*new_shape)
        self.sync_array()
        if self.rank == 0:
            self.array[self.i_time] = np.datetime64(time)
        self.i_time += 1
        self.comm.barrier()
