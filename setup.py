from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
# This line only needed if building with NumPy in Cython file.
from numpy import get_include
from os import system
from glob import glob
import os

setup_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
      Extension(# module name:
             'fv3gfs',
             # source file:
             ['fv3gfs.pyx'],
             include_dirs=[
                  get_include(), os.path.join(setup_dir, 'fms'), os.path.join(setup_dir, 'gfsphysics'),
                  os.path.join(setup_dir, 'io'), os.path.join(setup_dir, 'atmos_cubed_sphere'),
                  '/usr/lib/x86_64-linux-gnu/',
            ],
             library_dirs=[
                  '/opt/NCEPlibs/lib', '/usr/lib/gcc/x86_64-linux-gnu/9/',
                  '/usr/lib/x86_64-linux-gnu/'
            ],
             # other compile args for gcc
             extra_compile_args=['-fPIC', '-O3'],
             # other files to link to
             extra_link_args=[
                  'coupler_lib.o', 'physics_data.o', 'dynamics_data.o', 'atmos_model.o', 'atmos_cubed_sphere/libfv3core.a', 'io/libfv3io.a',
                  'gfsphysics/libgfsphys.a', 'stochastic_physics/libstochastic_physics.a', 'fms/libfms.a',
                  '/opt/NCEPlibs/lib/libnemsio_d.a', '/opt/NCEPlibs/lib/libbacio_4.a',
                  '/opt/NCEPlibs/lib/libsp_v2.0.2_d.a', '/opt/NCEPlibs/lib/libw3emc_d.a', '/opt/NCEPlibs/lib/libw3nco_d.a',
                  '-lgfortran', '-lpython3.7m', '-lmpi_mpifh', '-lmpi', '-lnetcdf', '-lnetcdff', '-fopenmp',
                  '-lmvec', '-lblas',
            ]
      )
]

setup(
      name = 'fv3gfs',
      cmdclass = {'build_ext': build_ext},
      # Needed if building with NumPy.
      # This includes the NumPy headers when compiling.
      include_dirs = [get_include()],
      ext_modules = cythonize(ext_modules)
)
