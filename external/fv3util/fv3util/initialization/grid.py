import dataclasses
from typing import Tuple, Dict
from ..constants import N_HALO
from .. import constants
from ..partitioner import TilePartitioner


@dataclasses.dataclass
class GridSizer:

    nx: int
    ny: int
    nz: int
    n_halo: int
    extra_dim_lengths: Dict[str, int]

    def get_origin(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_extent(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_shape(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        raise NotImplementedError()


class SubtileGridSizer(GridSizer):

    @classmethod
    def _from_tile_params(
        cls,
        nx_tile: int,
        ny_tile: int,
        nz: int,
        n_halo,
        extra_dim_lengths: Dict[str, int],
        layout: Tuple[int, int],
        tile_partitioner: TilePartitioner = None,
        tile_rank: int = 0,
    ):
        if tile_partitioner is None:
            tile_partitioner = TilePartitioner(layout)
        y_slice, x_slice = tile_partitioner.subtile_slice(
            tile_rank,
            [constants.Y_DIM, constants.X_DIM],
            [ny_tile, nx_tile],
            overlap=True
        )
        nx = x_slice.stop - x_slice.start
        ny = y_slice.stop - y_slice.start
        return cls(nx, ny, nz, extra_dim_lengths)

    @classmethod
    def from_namelist(cls, namelist: dict, tile_partitioner: TilePartitioner = None, tile_rank: int = 0):
        """Create a SubtileGridSizer from a Fortran namelist.
        
        Args:
            namelist: A namelist for the fv3gfs fortran model
            tile_partitioner (optional): a partitioner to use for segmenting the tile.
                By default, a TilePartitioner is used.
            tile_rank (optional): current rank on tile. Default is 0. Only matters if
                different ranks have different domain shapes. If tile_partitioner
                is a TilePartitioner, this argument does not matter.
        """
        layout = namelist["fv_core_nml"]["layout"]
        nx_tile = namelist["fv_core_nml"]["npx"] - 1
        ny_tile = namelist["fv_core_nml"]["npy"] - 1
        nz = namelist["fv_core_nml"]["npz"]  # this one is already on mid-levels
        return cls._from_tile_params(nx_tile, ny_tile, nz, N_HALO, {}, layout, tile_partitioner, tile_rank)

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
        return tuple(extents[dim] for dim in dims)

    def get_shape(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
        shape_dict = self.extra_dim_lengths.copy()
        # must pad non-interface variables to have the same shape as interface variables
        shape_dict.update(
            {
                constants.X_DIM: self.nx + 1 + 2 * N_HALO,
                constants.X_INTERFACE_DIM: self.nx + 1 + 2 * N_HALO,
                constants.Y_DIM: self.ny + 1 + 2 * N_HALO,
                constants.Y_INTERFACE_DIM: self.ny + 1 + 2 * N_HALO,
                constants.Z_DIM: self.nz + 1,
                constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return tuple(shape_dict[dim] for dim in dims)
