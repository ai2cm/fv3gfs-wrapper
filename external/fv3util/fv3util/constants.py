MASTER_RANK = 0
X_DIM = "x"
X_INTERFACE_DIM = "x_interface"
Y_DIM = "y"
Y_INTERFACE_DIM = "y_interface"
Z_DIM = "z"
Z_INTERFACE_DIM = "z_interface"
Z_SOIL_DIM = "z_soil"
X_DIMS = (X_DIM, X_INTERFACE_DIM)
Y_DIMS = (Y_DIM, Y_INTERFACE_DIM)
HORIZONTAL_DIMS = X_DIMS + Y_DIMS
INTERFACE_DIMS = (X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM)

WEST = 0
EAST = 1
NORTH = 2
SOUTH = 3
NORTHWEST = 4
NORTHEAST = 5
SOUTHWEST = 6
SOUTHEAST = 7
EDGE_BOUNDARY_TYPES = (NORTH, SOUTH, WEST, EAST)
CORNER_BOUNDARY_TYPES = (NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST)
BOUNDARY_TYPES = EDGE_BOUNDARY_TYPES + CORNER_BOUNDARY_TYPES
N_HALO_DEFAULT = 3
