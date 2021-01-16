import sys
import os
import tempfile
import io
import ctypes
import subprocess
import yaml
import unittest
import shutil
import fv3config
import fv3gfs.wrapper
from mpi4py import MPI

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")

openmpi_flags = [
    "--allow-run-as-root",
    "--oversubscribe",
    "--mca",
    "btl_vader_single_copy_mechanism",
    "none",
]


def run_unittest_script(filename, *args, n_processes=6):
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


def main(test_dir):
    rank = MPI.COMM_WORLD.Get_rank()
    with open(os.path.join(test_dir, "default_config.yml"), "r") as f:
        config = yaml.safe_load(f)
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
