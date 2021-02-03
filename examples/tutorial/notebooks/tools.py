import matplotlib.pyplot as plt
import numpy as np


from fv3gfs.util import (
    TilePartitioner, CubedSpherePartitioner, CubedSphereCommunicator, Quantity,
    X_DIM, Y_DIM, Z_DIM, X_INTERFACE_DIM, Y_INTERFACE_DIM
)


def get_X_Y(shape):
    """Get coordinate locations for plotting a global field as a flattened cube."""
    X = np.zeros([shape[0], shape[1] + 1, shape[2] + 1]) + np.arange(0, shape[1] + 1)[None, :, None]
    Y = np.zeros([shape[0], shape[1] + 1, shape[2] + 1]) + np.arange(0, shape[2] + 1)[None, None, :]
    # offset and rotate the data for each rank, with zero at the "center"
    for tile, shift_x, shift_y, n_rotations in [
        (1, 1, 0, 0), (2, 0, 1, -1), (3, 2, 0, 1), (4, -1, 0, 1), (5, 0, -1, 0)
    ]:
        X[tile, :, :] += shift_x * shape[1]
        Y[tile, :, :] += shift_y * shape[2]
        X[tile, :, :] = np.rot90(X[tile, :, :], n_rotations)
        Y[tile, :, :] = np.rot90(Y[tile, :, :], n_rotations)
    return X, Y


def plot_global(quantity, mpi_comm_world, vmin, vmax, layout=(1, 1)):
    """Plot a quantity globally on the root rank as a flattened cube."""
    cube = CubedSphereCommunicator(
        mpi_comm_world,
        CubedSpherePartitioner(
            TilePartitioner(layout)
        ),
    )
    assert quantity.dims == (Y_DIM, X_DIM), "example written to plot 2D fields"
    global_quantity = cube.gather(quantity)
    if global_quantity is not None:  # only on first rank
        X, Y = get_X_Y(global_quantity.extent)
        plt.figure(figsize=(9, 5.5))
        for tile in range(global_quantity.extent[0]):
            im = plt.pcolormesh(
                X[tile, :, :],
                Y[tile, :, :],
                global_quantity.view[tile, :, :].T,
                vmin=vmin,
                vmax=vmax,
        )
        plt.colorbar(im)