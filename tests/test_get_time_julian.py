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


class GetTimeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GetTimeTests, self).__init__(*args, **kwargs)
        self.mpi_comm = MPI.COMM_WORLD

    def setUp(self):
        pass

    def tearDown(self):
        self.mpi_comm.barrier()

    def test_get_time(self):
        state = fv3gfs.get_state(names=["time"])
        assert isinstance(state["time"], cftime.DatetimeJulian)


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    config = yaml.safe_load(open(os.path.join(test_dir, "default_config.yml"), "r"))
    config["namelist"]["coupler_nml"]["calendar"] = "julian"
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
