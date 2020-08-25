import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import pkgconfig

# This line only needed if building with NumPy in Cython file.
from numpy import get_include

PACKAGE_VERSION = "0.5.0"

fv3gfs_build_path_environ_name = "FV3GFS_BUILD_DIR"
make_command = os.environ.get("MAKE", "make")
package_dir = os.path.dirname(os.path.abspath(__file__))


class BuildDirectoryError(Exception):
    pass


relative_wrapper_build_filenames = [
    "lib/coupler_lib.o",
    "lib/physics_data.o",
    "lib/dynamics_data.o",
]


library_link_args = pkgconfig.libs("fv3").split()

mpi_flavor = os.environ.get("MPI", "openmpi")
if mpi_flavor == "openmpi":
    library_link_args += pkgconfig.libs("ompi-fort").split()
else:
    library_link_args += pkgconfig.libs("mpich-fort").split()

requirements = [
    "mpi4py",
    "cftime>=1.2.1",
    "xarray>=0.15.1",
    "netCDF4>=1.4.2",
    "numpy",
    "fv3util>=0.5.1",
    "pyyaml",
]

setup_requirements = ["cython", "numpy", "jinja2"]

test_requirements = []

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md", "r", encoding="utf-8") as history_file:
    history = history_file.read()

if fv3gfs_build_path_environ_name in os.environ:
    fv3gfs_build_path = os.environ(fv3gfs_build_path_environ_name)
else:
    fv3gfs_build_path = os.path.join(package_dir, "lib/external/FV3/")

wrapper_build_filenames = []
for relative_filename in relative_wrapper_build_filenames:
    wrapper_build_filenames.append(os.path.join(package_dir, relative_filename))

ext_modules = [
    Extension(  # module name:
        "fv3gfs._wrapper",
        # source file:
        ["lib/_wrapper.pyx"],
        libraries=["c", "m", ],
        include_dirs=[get_include()],
        extra_link_args=wrapper_build_filenames + library_link_args,
        depends=wrapper_build_filenames,
    )
]

setup(
    author="Vulcan Technologies LLC",
    author_email="jeremym@vulcan.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    name="fv3gfs-python",
    license="BSD license",
    long_description=readme + "\n\n" + history,
    cmdclass={"build_ext": build_ext},
    packages=["fv3gfs"],
    # Needed if building with NumPy.
    # This includes the NumPy headers when compiling.
    include_dirs=[get_include()],
    ext_modules=cythonize(ext_modules),
    url="https://github.com/VulcanClimateModeling/fv3gfs-python",
    version=PACKAGE_VERSION,
    zip_safe=False,
)
