import dataclasses
from typing import Tuple, Dict
from ..constants import N_HALO
from .. import constants
from ..partitioner import TilePartitioner


@dataclasses.dataclass
class DimensionSizer:

    nx: int
    ny: int
    nz: int
    extra_dim_lengths: Dict[str, int]

    def get_origin(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_extent(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_shape(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        raise NotImplementedError()


class SubtileGrid(Grid):

    @classmethod
    def from_tile_params(
        cls,
        nx_tile: int,
        ny_tile: int,
        nz: int,
        layout: Tuple[int, int],
        extra_dim_lengths: Dict[str, int],
    ):
        partitioner = TilePartitioner(layout)
        nx = partitioner.subtile_nx(nx_tile)
        ny = partitioner.subtile_ny(ny_tile)
        return cls(nx, ny, nz, extra_dim_lengths)

    @classmethod
    def from_namelist(cls, namelist):
        nx_tile = namelist["fv_core_nml"]["npx"] - 1
        ny_tile = namelist["fv_core_nml"]["npy"] - 1
        nz = namelist["fv_core_nml"]["npz"]  # this one is already on mid-levels
        layout = namelist["fv_core_nml"]["layout"]
        return cls.from_tile_params(nx_tile, ny_tile, nz, N_HALO, layout, {})

    @property
    def dim_extents(self) -> Dict[str, int]:
        return_dict = self.extra_dim_lengths.copy()
        return_dict.update(
            {
                constants.X_DIM: self.nx,
                constants.X_INTERFACE_DIM: self.nx + 1,
                constants.Y_DIM: self.ny,
                constants.Y_INTERFACE_DIM: self.ny + 1,
                constants.Z_DIM: self.nz,
                constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return return_dict

    def get_origin(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        return_list = [
            N_HALO if dim in constants.HORIZONTAL_DIMS else 0 for dim in dims
        ]
        return tuple(return_list)

    def get_extent(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        extents = self.dim_extents
        return tuple(extents[dim] for dim in self.dims)

    def get_shape(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        shape_dict = self.extra_dim_lengths.copy()
        # must pad non-interface variables to have the same shape as interface variables
        shape_dict.update(
            {
                constants.X_DIM: self.nx + 1 + 2 * N_HALO,
                constants.X_INTERFACE_DIM: self.nx + 1 + 2 * N_HALO,
                constants.Y_DIM: self.ny + 1 + 2 * N_HALO,
                constants.Y_INTERFACE_DIM: self.ny + 1 + 2 * N_HALO,
                constants.Z_DIM: self.nz + 1 + 2 * N_HALO,
                constants.Z_INTERFACE_DIM: self.nz + 1 + 2 * N_HALO,
            }
        )
        return tuple(shape_dict[dim] for dim in self.dims)
