import os
from setuptools import setup, find_namespace_packages
from distutils.extension import Extension

# Specify these build requirements in pyproject.toml
# https://www.python.org/dev/peps/pep-0518/
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import pkgconfig

# This line only needed if building with NumPy in Cython file.
from numpy import get_include

PACKAGE_VERSION = "0.6.0"

fv3gfs_build_path_environ_name = "FV3GFS_BUILD_DIR"
make_command = os.environ.get("MAKE", "make")
package_dir = os.path.dirname(os.path.abspath(__file__))


class BuildDirectoryError(Exception):
    pass


relative_wrapper_build_filenames = [
    "lib/coupler_lib.o",
    "lib/physics_data.o",
    "lib/dynamics_data.o",
    "lib/flagstruct_data.o",
]

wrapper_build_filenames = []
for relative_filename in relative_wrapper_build_filenames:
    wrapper_build_filenames.append(os.path.join(package_dir, relative_filename))

# order of library link args matters
# dependencies must be to the right of dependees
# https://stackoverflow.com/questions/45135/why-does-the-order-in-which-libraries-are-linked-sometimes-cause-errors-in-gcc
library_link_args = []
library_link_args.extend(wrapper_build_filenames)
library_link_args += pkgconfig.libs("fv3").split()

mpi_flavor = os.environ.get("MPI", "openmpi")
if mpi_flavor == "openmpi":
    library_link_args += pkgconfig.libs("ompi-fort").split()
else:
    library_link_args += ["-lmpich", "-lmpifort", "-lmpichcxx"]

# need to include math and c library
library_link_args += ["-lmvec", "-lc"]

requirements = [
    "mpi4py>=3.0.3",
    "cftime>=1.2.1",
    "xarray>=0.15.1",
    "netCDF4>=1.4.2",
    "numpy>=1.16",
    "pyyaml>=5",
    "pace-util>=0.7.0",
]

test_requirements = []

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md", "r", encoding="utf-8") as history_file:
    history = history_file.read()

if fv3gfs_build_path_environ_name in os.environ:
    fv3gfs_build_path = os.environ(fv3gfs_build_path_environ_name)
else:
    fv3gfs_build_path = os.path.join(package_dir, "lib/external/FV3/")


ext_modules = [
    Extension(  # module name:
        "fv3gfs.wrapper._wrapper",
        # source file:
        ["lib/_wrapper.pyx"],
        include_dirs=[get_include()],
        extra_link_args=library_link_args,
        depends=wrapper_build_filenames,
    )
]

setup(
    author="Vulcan Technologies LLC",
    author_email="jeremym@vulcan.com",
    python_requires=">=3.6",
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
    tests_require=test_requirements,
    extras_require={"examples": ["sklearn_json"]},
    name="fv3gfs-wrapper",
    license="BSD license",
    long_description=readme + "\n\n" + history,
    cmdclass={"build_ext": build_ext},
    packages=find_namespace_packages(include=["fv3gfs.*"]),
    package_data={"fv3gfs.wrapper": ["*.json"]},
    # Needed if building with NumPy.
    # This includes the NumPy headers when compiling.
    include_dirs=[get_include()],
    ext_modules=cythonize(ext_modules),
    url="https://github.com/VulcanClimateModeling/fv3gfs-wrapper",
    version=PACKAGE_VERSION,
    zip_safe=False,
)
