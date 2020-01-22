import xarray as xr
import numpy as np
import pickle

CF_TO_RESTART_MAP = {
    'specific_humidity': 'sphum', 
    'air_temperature': 'T', 
}

RESTART_TO_CF_MAP = dict(zip(CF_TO_RESTART_MAP.values(), CF_TO_RESTART_MAP.keys()))

def dump(state, f):
    output = [{key: val.to_dict() for key, val in state_i.items()}
              for state_i in state]
    pickle.dump(output, f)


def load(f):
    output = pickle.load(f)
    return [{key: xr.DataArray.from_dict(val) for key, val in state_i.items()} for state_i in output]


class ZarrVariableWriter:
    def __init__(self, comm, group, name):
        self.idx = 0
        self.comm = comm
        self.group = group
        self.name = name
        self.array = None
    
    @property
    def rank(self):
        return self.comm.Get_rank()

    @property
    def size(self):
        return self.comm.Get_size()

    def _init_zarr(self, array):
        if self.rank == 0:
            self._init_zarr_root(array)
        self.array = self.comm.bcast(self.array, root=0)

    def _init_zarr_root(self, array):
        shape = (1, self.size) + array.shape
        chunks = (1, 1) + array.shape
        self.array = self.group.create_dataset(self.name, shape=shape,
        dtype=array.dtype, chunks=chunks)

    def set_dims(self, dims):
        self.array.attrs['_ARRAY_DIMENSIONS'] = dims

    def append(self, array):

        if self.array is None:
            self._init_zarr(array)
            self.set_dims(['time', 'rank'] + list(array.dims))

        if self.idx >= self.array.shape[0]:
            new_shape = (self.idx + 1, self.size) + self.array.shape[2:]
            self.array.resize(*new_shape)
        

        try:
            self.array[self.idx, self.rank, ...] = np.asarray(array)
        except Error as e:
            logger.critical("Exception Raised on rank", self.rank)
            raise e
        self.idx += 1


