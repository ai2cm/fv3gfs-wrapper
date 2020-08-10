"""Example usage:

$ mpirun -n 6 --allow-run-as-root \
     --oversubscribe \
     --mca btl_vader_single_copy_mechanism none \
     python3 -m mpi4py test_get_time.py noleap

Note the argument specifying the calendar type at the end of the command
is required.  Valid calendars are:
- julian
- noleap
- thirty_day
 """
import sys
import unittest
import yaml
import os
import shutil
import cftime
import fv3config
import fv3gfs
from mpi4py import MPI
from util import redirect_stdout

test_dir = os.path.dirname(os.path.abspath(__file__))

CFTIME_TYPES = {
    "julian": cftime.DatetimeJulian,
    "noleap": cftime.DatetimeNoLeap,
    "thirty_day": cftime.Datetime360Day,
}


class GetTimeTests(unittest.TestCase):
    DATE_TYPE = None

    def __init__(self, *args, **kwargs):
        super(GetTimeTests, self).__init__(*args, **kwargs)
        self.mpi_comm = MPI.COMM_WORLD

    def setUp(self):
        pass

    def tearDown(self):
        self.mpi_comm.barrier()

    def test_get_time(self):
        state = fv3gfs.get_state(names=["time"])
        assert isinstance(state["time"], self.DATE_TYPE)


def set_calendar_type():
    """Required for setting the date type for GetTimeTests with a command line
    argument.

    See https://stackoverflow.com/questions/11380413/python-unittest-passing-arguments.
    """
    if len(sys.argv) != 2:
        raise ValueError(
            "test_get_time.py requires a single calendar argument "
            "be passed through the command line."
        )
    calendar = sys.argv.pop()
    GetTimeTests.DATE_TYPE = CFTIME_TYPES[calendar]
    return calendar


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    calendar = set_calendar_type()
    config = yaml.safe_load(open(os.path.join(test_dir, "default_config.yml"), "r"))
    config["namelist"]["coupler_nml"]["calendar"] = calendar
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
            fv3gfs.initialize()
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
