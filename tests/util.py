import sys
import os
import tempfile
import io
import ctypes
import subprocess
import yaml
import unittest
import shutil
import numpy as np
import fv3config
import fv3gfs.wrapper
from copy import deepcopy
from mpi4py import MPI

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
base_dir = os.path.dirname(os.path.realpath(__file__))


def run_unittest_script(script_name, *args, n_processes=6):
    filename = os.path.join(base_dir, script_name)
    python_args = ["python3", "-m", "mpi4py", filename] + list(args)
    subprocess.check_call(["mpirun", "-n", str(n_processes)] + python_args)


def redirect_stdout(filename):
    """
    Context manager for temporarily redirecting sys.stdout to another file.

    Behaves similarly to `contextlib.redirect_stdout`, but will also apply the
    redirection to code in compiled shared libraries.

    Usage:
        with redirect_stdout(log_filename):
            # output will be redirected to log file
            do_stuff()
    """
    return StdoutRedirector(filename)


class StdoutRedirector(object):

    __doc__ = redirect_stdout.__doc__

    def __init__(self, filename):
        self.stream = open(filename, "wb")
        self._filename = filename
        self._stdout_file_descriptor = sys.stdout.fileno()
        self._saved_stdout_file_descriptor = os.dup(self._stdout_file_descriptor)
        self._temporary_stdout = tempfile.TemporaryFile(mode="w+b")

    def __enter__(self):
        # Redirect the stdout file descriptor to point at our temporary file
        self._redirect_stdout(self._temporary_stdout.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        # Set the stdout file descriptor back to what it was when we started
        self._redirect_stdout(self._saved_stdout_file_descriptor)
        # Write contents of temporary file to the output file
        self._temporary_stdout.flush()
        self._temporary_stdout.seek(0, io.SEEK_SET)
        self.stream.write(self._temporary_stdout.read())
        # Close our temporary file and remove the duplicate file descriptor
        self._temporary_stdout.close()
        os.close(self._saved_stdout_file_descriptor)

    def _redirect_stdout(self, to_file_descriptor):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make self._stdout_file_descriptor point to the same file as to_file_descriptor
        # This redirects stdout to the file at the file descriptor level, which C/Fortran also obeys
        os.dup2(to_file_descriptor, self._stdout_file_descriptor)
        # Create a new sys.stdout for Python that points to the redirected file descriptor
        sys.stdout = io.TextIOWrapper(os.fdopen(self._stdout_file_descriptor, "wb"))


def main(test_dir, config):
    rank = MPI.COMM_WORLD.Get_rank()
    rundir = os.path.join(test_dir, "rundir")
    if rank == 0:
        if os.path.isdir(rundir):
            shutil.rmtree(rundir)
        fv3config.write_run_directory(config, rundir)
    MPI.COMM_WORLD.barrier()
    original_path = os.getcwd()
    os.chdir(rundir)
    try:
        with redirect_stdout(os.devnull):
            fv3gfs.wrapper.initialize()
            MPI.COMM_WORLD.barrier()
        if rank != 0:
            kwargs = {"verbosity": 0}
        else:
            kwargs = {"verbosity": 2}
        unittest.main(**kwargs)
    finally:
        os.chdir(original_path)
        if rank == 0:
            shutil.rmtree(rundir)


def get_default_config():
    with open(os.path.join(base_dir, "default_config.yml"), "r") as f:
        return yaml.safe_load(f)


def get_current_config():
    with open("fv3config.yml") as f:
        return yaml.safe_load(f)


def generate_data_dict(properties):
    return {entry["name"]: entry for entry in properties}


def replace_state_with_random_values(names):
    old_state = fv3gfs.wrapper.get_state(names=names)
    replace_state = deepcopy(old_state)
    for name, quantity in replace_state.items():
        quantity.view[:] = np.random.uniform(size=quantity.extent)
    fv3gfs.wrapper.set_state(replace_state)
    return replace_state


def get_state_single_variable(name):
    return fv3gfs.wrapper.get_state([name])[name].view[:]
