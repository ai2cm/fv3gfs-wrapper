import os
import shutil
import sys
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# This line only needed if building with NumPy in Cython file.
from numpy import get_include


PACKAGE_VERSION = "0.5.0"


class BuildDirectoryError(Exception):
    pass


def find_coupler_lib(dir_="lib"):
    path = os.path.join(dir_, "libcoupler.a")
    if os.path.exists(path):
        return ["-lcoupler", f"-L{dir_}"]
    else:
        raise BuildDirectoryError(f"File {path} is missing, first run make in {dir_}")


def find_mpi_libraries():
    mpi_flavor = os.environ.get("MPI", "openmpi")
    if mpi_flavor == "openmpi":
        mpi_fortran_lib = "-lmpi_mpifh"
    else:
        mpi_fortran_lib = "-lmpifort"

    return [mpi_fortran_lib]


def find_python_library():
    return ["-lpython3." + str(sys.version_info.minor) + "m"]


def find_fv3_executable():
    if fv3gfs_build_path_environ_name in os.environ:
        fv3gfs_build_path = os.environ(fv3gfs_build_path_environ_name)
    else:
        fv3gfs_build_path = os.path.join(package_dir, "lib/external/FV3/")
    return os.path.join(fv3gfs_build_path, "fv3.exe")


fv3gfs_build_path_environ_name = "FV3GFS_BUILD_DIR"
make_command = os.environ.get("MAKE", "make")
package_dir = os.path.dirname(os.path.abspath(__file__))


requirements = [
    "cftime>=1.2.1",
    "xarray>=0.15.1",
    "netCDF4>=1.4.2",
    "numpy",
    f"fv3util>=0.5.1",
]

setup_requirements = ["cython", "numpy", "jinja2"]

test_requirements = []

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md", "r", encoding="utf-8") as history_file:
    history = history_file.read()


extra_link_args = [
    "-lFMS",
    "-lesmf",
    "-lgfortran",
    "-lmpi",
    "-lnetcdf",
    "-lnetcdff",
    "-fopenmp",
    "-lmvec",
    "-lblas",
    "-lc",
    "-lrt",
]
extra_link_args += find_coupler_lib()
extra_link_args += find_mpi_libraries()
extra_link_args += find_python_library()

# copy2 preserves executable flag
shutil.copy2(find_fv3_executable(), os.path.join(package_dir, "fv3.exe"))

ext_modules = [
    Extension(  # module name:
        "fv3gfs._wrapper",
        # source file:
        ["lib/_wrapper.pyx"],
        include_dirs=[get_include()],
        extra_link_args=extra_link_args,
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
