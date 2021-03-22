"""Example usage:

$ mpirun -n 6  \
     python3 -m mpi4py test_get_time.py noleap

Note the argument specifying the calendar type at the end of the command
is required.  Valid calendars are:
- julian
- noleap
- thirty_day
 """
import sys
import unittest
import os
import cftime
import fv3gfs.wrapper
from mpi4py import MPI
from util import get_default_config, main

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
        state = fv3gfs.wrapper.get_state(names=["time"])
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
    calendar = set_calendar_type()
    config = get_default_config()
    config["namelist"]["coupler_nml"]["calendar"] = calendar
    main(test_dir, config)
