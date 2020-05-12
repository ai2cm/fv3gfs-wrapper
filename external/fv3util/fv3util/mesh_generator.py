from typing import Tuple
import math
import numpy as np

from constants import (
    N_HALO_DEFAULT,
)

class MeshGenerator:

    RADIUS = 6.3712e+6
    RIGHT_HAND = False
    DEFAULT_INITIALIZATION_VALUE = np.NaN

    def __init__(
        self,
        grid_type: int,           # type of grid (0 = equal-distance Gnomonic)
        shape: Tuple[int, int],   # number of nodes on tile in x- and y-direction
        ntiles: int,              # number of tiles
        ng: int,                  # number of halo-points
        shift_fac=18.0            # shift of prime meridian
    ):
        """Create an mesh generator object.
        """
        assert grid_type == 0, "Only true equal-distance Gnomonic grid supported"
        
        self.grid_type = grid_type
        self.ntiles = ntiles
        npx, npy = shape
        self.npx = npx
        self.npy = npy
        self.ng  = ng
        self.shift_fac = shift_fac

        self.lon, self.lat = self.__generate_global_grid(grid_type, 2, ntiles, npx, npy, ng, shift_fac=shift_fac)

    @classmethod
    def from_namelist(cls, namelist):
        """Initialize a MeshGenerator from a FV3GFS Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        grid_type = 0
        ntiles = 6
        npx = namelist["fv_core_nml"]["npx"]
        npy = namelist["fv_core_nml"]["npy"]
        ng = N_HALO_DEFAULT
        return cls(grid_type, [npx, npy], ntiles, ng)

    @classmethod
    def from_name(cls, name):
        """Initialize a MeshGenerator from a cubed-sphere grid name.

        Args:
            name (str): the cubed-sphere grid name (e.g. c48)
        """
        assert name[0] == "c", "Only cubed-sphere grids supported"
        grid_type = 0
        ntiles = 6
        npx = npy = int(name[1:]) + 1
        ng = N_HALO_DEFAULT
        return cls(grid_type, [npx, npy], ntiles, ng)

    def extract_local_grid(
        self,
        tile: int,
        subdomain: Tuple[int, int, int, int],
        zero=False
    ):
        """Extract subtile from global mesh.

        Args:
            tile (int): the tile number to extract
            subdomain (Tuple): subtile to extract
            zero: initialize halo values to zero?
        """
        i_start, i_end = subdomain[0], subdomain[1]
        j_start, j_end = subdomain[2], subdomain[3]
        ng = self.ng

        shape = (i_end-i_start+2*ng, j_end-j_start+2*ng, self.ndims)
        grid = self.__empty_array(shape, zero=zero)

        lon[ng:-ng, ng:-ng, :] = self.lon[i_start:i_end, j_start:j_end, tile]
        lat[ng:-ng, ng:-ng, :] = self.lat[i_start:i_end, j_start:j_end, tile]

        return lon, lat

    def __empty_array(self, shape, dtype=float, order='F', zero=False):
        array = np.empty(shape, dtype, order)
        if zero:
            array[:] = 0.0
        elif self.DEFAULT_INITIALIZATION_VALUE is not None:
            array[:] = self.DEFAULT_INITIALIZATION_VALUE
        return array

    def __generate_global_grid(self, grid_type, ndims, ntiles, npx, npy, ng, shift_fac=0.0):

        grid_global = self.__empty_array((npx + 2*ng, npy + 2*ng, ndims, ntiles))

        lon, lat = self.__gnomonic_grids(grid_type, npx-1)
        
        grid_global[ng:npx+ng, ng:npy+ng, 0, 0] = lon
        grid_global[ng:npx+ng, ng:npy+ng, 1, 0] = lat

        grid_global = self.__mirror_grid(grid_global, ng, npx, npy, ndims, ntiles)

        # Note: shift corner away from Japan
        #       this will result in the corner close to the east coast of China
        for n in range(0, ntiles):
            for j in range(0, npy):
                for i in range(0, npx):
                    if shift_fac > 1.0e-4:
                        grid_global[ng + i, ng + j, 0, n] -= np.pi / shift_fac
                    if grid_global[ng + i, ng + j, 0, n] < 0.0:
                        grid_global[ng + i, ng + j, 0, n] += 2.0 * np.pi
                    if np.abs(grid_global[ng + i, ng + j, 0, n]) < 1.0e-10:
                        grid_global[ng + i, ng + j, 0, n] = 0.0
                    if np.abs(grid_global[ng + i, ng + j, 1, n]) < 1.0e-10:
                        grid_global[ng + i, ng + j, 1, n] = 0.0
        
        grid_global[ng, ng:npy+ng, :, 1] = grid_global[npx-1+ng, ng:npy+ng, :, 0]
        grid_global[ng, ng:npy+ng, :, 2] = grid_global[npx+ng-1:ng-1:-1, npy-1+ng, : ,0]
        grid_global[ng:npx+ng, npy-1+ng, :, 4] = grid_global[ng, npy+ng-1:ng-1:-1, :, 0]
        grid_global[ng:npx+ng, npy-1+ng, :, 5] = grid_global[ng:npx+ng, ng, :, 0]

        grid_global[ng:npx+ng, ng, :, 2] = grid_global[ng:npx+ng, npy-1+ng, :, 1]
        grid_global[ng:npx+ng, ng, :, 3] = grid_global[npx-1+ng, npy+ng-1:ng-1:-1, :, 1]
        grid_global[npx-1+ng, ng:npy+ng, :, 5] = grid_global[npx+ng-1:ng-1:-1, ng, :, 1]

        grid_global[ng, ng:npy+ng, :, 3] = grid_global[npx-1+ng, ng:npy+ng, :, 2]
        grid_global[ng, ng:npy+ng, :, 4] = grid_global[npx+ng-1:ng-1:-1, npy-1+ng, :, 2]

        grid_global[npx-1+ng, ng:npy+ng, :, 2] = grid_global[ng, ng:npy+ng, :, 3]
        grid_global[ng:npx+ng, ng, :, 4] = grid_global[ng:npx+ng, npy-1+ng, :, 3]
        grid_global[ng:npx+ng, ng, :, 5] = grid_global[npx-1+ng, npy+ng-1:ng-1:-1, :, 3]

        grid_global[ng, ng:npy+ng, :, 5] = grid_global[npx-1+ng, ng:npy+ng, :, 4]

        return grid_global[:, :, 0, :], grid_global[:, :, 1, :]

    def __mirror_grid(self, grid_global, ng, npx, npy, ndims, ntiles):

        # first fix base region
        nreg = 0
        for j in range(0, math.ceil(npy / 2)):
            for i in range(0, math.ceil(npx / 2)):
                x1 = 0.25 * (np.abs(grid_global[ng + i        , ng + j        , 0, nreg]) +
                             np.abs(grid_global[ng + npx-(i+1), ng + j        , 0, nreg]) +
                             np.abs(grid_global[ng + i        , ng + npy-(j+1), 0, nreg]) +
                             np.abs(grid_global[ng + npx-(i+1), ng + npy-(j+1), 0, nreg]))
                grid_global[ng + i        , ng + j        , 0, nreg] = np.copysign(x1, grid_global[ng + i        , ng + j        , 0, nreg])
                grid_global[ng + npx-(i+1), ng + j        , 0, nreg] = np.copysign(x1, grid_global[ng + npx-(i+1), ng + j        , 0, nreg])
                grid_global[ng + i        , ng + npy-(j+1), 0, nreg] = np.copysign(x1, grid_global[ng + i        , ng + npy-(j+1), 0, nreg])
                grid_global[ng + npx-(i+1), ng + npy-(j+1), 0, nreg] = np.copysign(x1, grid_global[ng + npx-(i+1), ng + npy-(j+1), 0, nreg])

                y1 = 0.25 * (np.abs(grid_global[ng + i        , ng + j        , 1, nreg]) +   
                             np.abs(grid_global[ng + npx-(i+1), ng + j        , 1, nreg]) +
                             np.abs(grid_global[ng + i        , ng + npy-(j+1), 1, nreg]) +
                             np.abs(grid_global[ng + npx-(i+1), ng + npy-(j+1), 1, nreg]))
                grid_global[ng + i        , ng + j        , 1, nreg] = np.copysign(y1, grid_global[ng + i        , ng + j        , 1, nreg])
                grid_global[ng + npx-(i+1), ng + j        , 1, nreg] = np.copysign(y1, grid_global[ng + npx-(i+1), ng + j        , 1, nreg])
                grid_global[ng + i        , ng + npy-(j+1), 1, nreg] = np.copysign(y1, grid_global[ng + i        , ng + npy-(j+1), 1, nreg])
                grid_global[ng + npx-(i+1), ng + npy-(j+1), 1, nreg] = np.copysign(y1, grid_global[ng + npx-(i+1), ng + npy-(j+1), 1, nreg])
             
                # force dateline/greenwich-meridion consitency
                if npx % 2 != 0:
                    if i == ng + (npx - 1) // 2:
                        grid_global[ng + i, ng + j        , 0, nreg] = 0.0
                        grid_global[ng + i, ng + npy-(j+1), 0, nreg] = 0.0

        for nreg in range(1, ntiles):
            for j in range(0, npy):
                for i in range(0, npx):
                    x1 = grid_global[ng + i, ng + j, 0, 0]
                    y1 = grid_global[ng + i, ng + j, 1, 0]
                    z1 = self.RADIUS

                    if nreg == 1:
                        ang = -90.0
                        x2, y2, z2 = self.__rot_3d(3, [x1, y1, z1], ang, degrees=True, convert=True)
                    elif nreg == 2:
                        ang = -90.0
                        x2, y2, z2 = self.__rot_3d(3, [x1, y1, z1], ang, degrees=True, convert=True)
                        ang = 90.0
                        x2, y2, z2 = self.__rot_3d(1, [x2, y2, z2], ang, degrees=True, convert=True)
                        # force North Pole and dateline/Greenwich-Meridian consistency
                        if npx % 2 != 0:
                            if i == (npx - 1) // 2 and i == j:
                                x2 = 0.0
                                y2 = np.pi / 2.0
                            if j == (npy - 1) // 2 and i < (npx - 1) // 2:
                                x2 = 0.0
                            if j == (npy - 1) // 2 and i > (npx - 1) // 2:
                                x2 = np.pi
                    elif nreg == 3:
                        ang = -180.0
                        x2, y2, z2 = self.__rot_3d(3, [x1, y1, z1], ang, degrees=True, convert=True)
                        ang = 90.0
                        x2, y2, z2 = self.__rot_3d(1, [x2, y2, z2], ang, degrees=True, convert=True)
                        # force dateline/Greenwich-Meridian consistency
                        if npx % 2 != 0:
                            if j == (npy - 1) // 2:
                                x2 = np.pi
                    elif nreg == 4:
                        ang = 90.0
                        x2, y2, z2 = self.__rot_3d(3, [x1, y1, z1], ang, degrees=True, convert=True)
                        ang = 90.0
                        x2, y2, z2 = self.__rot_3d(2, [x2, y2, z2], ang, degrees=True, convert=True)
                    elif nreg == 5:
                        ang = 90.0
                        x2, y2, z2 = self.__rot_3d(2, [x1, y1, z1], ang, degrees=True, convert=True)
                        ang = 0.0
                        x2, y2, z2 = self.__rot_3d(3, [x2, y2, z2], ang, degrees=True, convert=True)
                        # force South Pole and dateline/Greenwich-Meridian consistency
                        if npx % 2 != 0:
                            if i == (npx - 1) // 2 and i == j:
                                x2 = 0.0
                                y2 = -np.pi / 2.0
                            if i == (npx - 1) // 2 and j > (npy - 1) // 2:
                                x2 = 0.0
                            if i == (npx - 1) // 2 and j < (npy - 1) // 2:
                                x2 = np.pi
                    
                    grid_global[ng + i, ng + j, 0, nreg] = x2
                    grid_global[ng + i, ng + j, 1, nreg] = y2
    
        return grid_global

    def __rot_3d(self, axis, p, angle, degrees=False, convert=False):

        if convert:
            p1 = self.__spherical_to_cartesian(p)
        else:
            p1 = p
        
        if degrees:
            angle = np.deg2rad(angle)

        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 1:
            x2 = p1[0]
            y2 = c*p1[1] + s*p1[2]
            z2 = -s*p1[1] + c*p1[2]
        elif axis == 2:
            x2 = c*p1[0] - s*p1[2]
            y2 = p1[1]
            z2 = s*p1[0] + c*p1[2]
        elif axis == 3:
            x2 = c*p1[0] + s*p1[1]
            y2 = -s*p1[0] + c*p1[1]
            z2 = p1[2]
        else:
            assert False, "axis must be in [1,2,3]"
        
        if convert:
            p2 = self.__cartesian_to_spherical([x2, y2, z2])
        else:
            p2 = [x2, y2, z2]

        return p2

    def __spherical_to_cartesian(self, p):
        lon, lat, r = p
        x = r * np.cos(lon) * np.cos(lat)
        y = r * np.sin(lon) * np.cos(lat)
        if self.RIGHT_HAND:
            z = r * np.sin(lat)
        else:
            z = -r * np.sin(lat)
        return [x, y, z]

    def __cartesian_to_spherical(self, p):
        x, y, z = p
        r = np.sqrt(x*x + y*y + z*z)
        if np.abs(x) + np.abs(y) < 1.0e-10:
            lon = 0.0
        else:
            lon = np.arctan2(y,x)
        if self.RIGHT_HAND:
            lat = np.arcsin(z/r)
        else:
            lat = np.arccos(z/r) - np.pi/2.0
        return [lon, lat, r]

    def __gnomonic_grids(self, grid_type, im):
        if grid_type == 0:
            lon, lat = self.__gnomonic_ed(im)
        else:
            assert False, "Only true equal-distance Gnomonic grid supported"

        if grid_type < 3:
            lon, lat = self.__symm_ed(im, lon, lat)
            lon -= np.pi

        return lon, lat

    def __gnomonic_ed(self, im):
        rsq3 = 1.0 / np.sqrt(3.0)
        alpha = np.arcsin( rsq3 )

        dely = 2.0 * alpha / float(im)

        lon = self.__empty_array( (im+1, im+1) )
        lat = self.__empty_array( (im+1, im+1) )
        pp = self.__empty_array( (3, im+1, im+1) )

        for j in range(0, im+1):
            lon[0,  j] = 0.75 * np.pi              # West edge
            lon[im, j] = 1.25 * np.pi              # East edge
            lat[0,  j] = -alpha + dely * float(j)  # West edge
            lat[im, j] = lat[0, j]                  # East edge

        # Get North-South edges by symmetry
        for i in range(1, im):
            lon[i, 0], lat[i, 0] = self.__mirror_latlon(lon[0,  0 ], lat[0,  0 ],
                                                        lon[im, im], lat[im, im],
                                                        lon[0,  i ], lat[0,  i ])
            lon[i, im] = lon[i, 0]
            lat[i, im] = -lat[i, 0]

        # set 4 corners
        pp[:, 0, 0] = self.__latlon2xyz(lon[0, 0], lat[0, 0])
        pp[:, im, 0] = self.__latlon2xyz(lon[im, 0], lat[im, 0])
        pp[:, 0, im] = self.__latlon2xyz(lon[0, im], lat[0, im])
        pp[:, im, im] = self.__latlon2xyz(lon[im, im], lat[im, im])

        # map edges on the sphere back to cube: intersection at x = -rsq3
        i = 0
        for j in range(1, im):
            pp[:, i, j] = self.__latlon2xyz(lon[i, j], lat[i, j])
            pp[1, i, j] = -pp[1, i, j] * rsq3 / pp[0, i, j]
            pp[2, i, j] = -pp[2, i, j] * rsq3 / pp[0, i, j]

        j = 0
        for i in range(1, im):
            pp[:, i, j] = self.__latlon2xyz(lon[i, j], lat[i, j])
            pp[1, i, j] = -pp[1, i, j] * rsq3 / pp[0, i, j]
            pp[2, i, j] = -pp[2, i, j] * rsq3 / pp[0, i, j]
        
        for j in range(0, im+1):
            for i in range(0, im+1):
                pp[0, i, j] = -rsq3

        for j in range(1, im+1):
            for i in range(1, im+1):
                # copy y-z face of the cube along j=0
                pp[1, i, j] = pp[1, i, 0]
                # copy along i=0
                pp[2, i, j] = pp[2, 0, j]

        pp, lon, lat = self.__cart_to_latlon( im+1, pp, lon, lat)

#        # compute great-circle distance "resolution" along the face edge
#        p1 = (lon[0, 0], lat[0, 0])
#        p2 = (lon[1, 0], lat[1, 0])
#        dist = self.__great_circle_dist(p1, p2, self.RADIUS)
#        print("Grid distance at face edge [km] = {}".format(dist))

        return lon, lat

    def __symm_ed(self, im, lon, lat):

        for j in range(1, im+1):
            for i in range(1, im):
                lon[i, j] = lon[i, 0]

        # make grid symmetrical to i = im/2
        for j in range(0, im+1):
            for i in range(0, im//2):
                ip = im - i
                avg = 0.5 * (lon[i, j] - lon[ip, j])
                lon[i, j] = avg + np.pi
                lon[ip, j] = np.pi - avg
                avg = 0.5 * (lat[i, j] + lat[ip, j])
                lat[i, j] = avg
                lat[ip, j] = avg
       
        # make grid symmetrical to j = im/2
        for j in range(0, im // 2):
            jp = im - j
            for i in range(1, im):
                avg = 0.5 * (lon[i, j] + lon[i, jp])
                lon[i, j] = avg
                lon[i, jp] = avg
                avg = 0.5 * (lat[i, j] - lat[i, jp])
                lat[i, j] = avg
                lat[i, jp] = -avg

        return lon, lat

    def __mirror_latlon(self, lon1, lat1, lon2, lat2, lon0, lat0):

        p0 = self.__latlon2xyz(lon0, lat0)
        p1 = self.__latlon2xyz(lon1, lat1)
        p2 = self.__latlon2xyz(lon2, lat2)
        nb = self.__vect_cross(p1, p2)

        pdot = np.sqrt(nb[0]**2 + nb[1]**2 + nb[2]**2)
        nb = nb / pdot

        pdot = p0[0]*nb[0] + p0[1]*nb[1] + p0[2]*nb[2]
        pp = p0 - 2.0 * pdot * nb

        lon3 = self.__empty_array( (1, 1) )
        lat3 = self.__empty_array( (1, 1) )
        pp3 = self.__empty_array( (3, 1, 1) )
        pp3[:, 0, 0] = pp
        self.__cart_to_latlon(1, pp3, lon3, lat3)
        
        return lon3[0, 0], lat3[0, 0]

    def __latlon2xyz(self, lon, lat):
        """map (lon, lat) to (x, y, z)"""

        e1 = np.cos(lat) * np.cos(lon)
        e2 = np.cos(lat) * np.sin(lon)
        e3 = np.sin(lat)

        return [e1, e2, e3]

    def __cart_to_latlon(self, im, q, xs, ys):
        """map (x, y, z) to (lon, lat)"""

        esl = 1.0e-10

        for i in range(im):
            for j in range(im):
                p = q[:, i, j]
                dist = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                p = p / dist

                if np.abs(p[0]) + np.abs(p[1]) < esl:
                    lon = 0.0
                else:
                    lon = np.arctan2(p[1], p[0])  # range [-pi, pi]

                if lon < 0.0:
                    lon = 2.0 * np.pi + lon

                lat = np.arcsin(p[2])
            
                xs[i, j] = lon
                ys[i, j] = lat

                q[:, i, j] = p

        return q, xs, ys

    def __great_circle_dist(self, p1, p2, radius=None):

        beta = np.arcsin( np.sqrt( np.sin( (p1[1] - p2[1])/2.0)**2 + np.cos(p1[1])*np.cos(p2[1]) *
                                   np.sin( (p1[0] - p2[0])/2.0)**2 ) ) * 2.0

        if radius is None:
            return beta
        else:
            return beta * radius

    def __vect_cross(self, p1, p2):
        return [p1[1]*p2[2] - p1[2]*p2[1], p1[2]*p2[0] - p1[0]*p2[2], p1[0]*p2[1] - p1[1]*p2[0]]

