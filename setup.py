from glob import glob
import subprocess
import os
import shutil
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
# This line only needed if building with NumPy in Cython file.
from numpy import get_include


fv3gfs_build_path_environ_name = 'FV3GFS_BUILD_DIR'
make_command = os.environ.get('MAKE', 'make')
package_dir = os.path.dirname(os.path.abspath(__file__))


class BuildDirectoryError(Exception):
    pass


relative_wrapper_build_filenames = ['lib/coupler_lib.o', 'lib/physics_data.o', 'lib/dynamics_data.o', ]

relative_fv3gfs_build_filenames = [
    'atmos_model.o', 'atmos_cubed_sphere/libfv3core.a', 'io/libfv3io.a',
    'gfsphysics/libgfsphys.a', 'stochastic_physics/libstochastic_physics.a', 'fms/libfms.a',
    '/opt/NCEPlibs/lib/libnemsio_d.a', '/opt/NCEPlibs/lib/libbacio_4.a',
    '/opt/NCEPlibs/lib/libsp_v2.0.2_d.a', '/opt/NCEPlibs/lib/libw3emc_d.a', '/opt/NCEPlibs/lib/libw3nco_d.a',
]

library_link_args = [
    '-lgfortran', '-lpython3.7m', '-lmpi_mpifh', '-lmpi', '-lnetcdf', '-lnetcdff', '-fopenmp',
    '-lmvec', '-lblas',
]

requirements = [
    'xarray>=0.13.0',
    'netCDF4>=1.4.2',
    'numpy'
]

setup_requirements = ['cython', 'numpy', 'jinja2']

test_requirements = []

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

if fv3gfs_build_path_environ_name in os.environ:
    fv3gfs_build_path = os.environ(fv3gfs_build_path_environ_name)
else:
    fv3gfs_build_path = os.path.join(package_dir, 'lib/FV3/sorc/fv3gfs.fd/FV3/')

fortran_build_filenames = []
for relative_filename in relative_fv3gfs_build_filenames:
    fortran_build_filenames.append(os.path.join(fv3gfs_build_path, relative_filename))

wrapper_build_filenames = []
for relative_filename in relative_wrapper_build_filenames:
    wrapper_build_filenames.append(os.path.join(package_dir, relative_filename))

# make library dependencies before checking if they exist
subprocess.check_call([make_command], cwd=os.path.join(package_dir, 'lib'))

for filename in fortran_build_filenames:
    if not os.path.isfile(filename):
        raise BuildDirectoryError(f'File {filename} is missing, does {fv3gfs_build_path_environ_name} contain FV3GFS build artifacts?')

# copy2 preserves executable flag
shutil.copy2(os.path.join(fv3gfs_build_path, 'fv3.exe'), os.path.join(package_dir, 'fv3.exe'))

ext_modules = [
    Extension(# module name:
        'fv3gfs._wrapper',
        # source file:
        ['lib/_wrapper.pyx'],
        include_dirs=[
            get_include(),
        ],
        extra_link_args= wrapper_build_filenames + fortran_build_filenames + library_link_args,
        depends=fortran_build_filenames + wrapper_build_filenames,
    )
]

setup(
    author="Vulcan Technologies, LLC",
    author_email='jeremym@vulcan.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    name='fv3gfs',
    license="BSD license",
    long_description=readme + '\n\n' + history,
    cmdclass={'build_ext': build_ext},
    packages=['fv3gfs'],
    # Needed if building with NumPy.
    # This includes the NumPy headers when compiling.
    include_dirs=[get_include()],
    ext_modules=cythonize(ext_modules),
    url='https://github.com/VulcanClimateModeling/fv3gfs',
    version='0.1.0',
    zip_safe=False,
)
