# Configuration file for ipcluster.
# See https://github.com/erdc/ipython_profile/blob/master/ipcluster_config.py
# for a full set of available configuration options.

## Whether to log to a file
c.BaseParallelApplication.log_to_file = True

## The command line arguments to pass to mpiexec.
c.MPILauncher.mpi_args = []

## The mpiexec command to use in starting the process.
c.MPILauncher.mpi_cmd = ["mpiexec"]

## command-line args to pass to ipcontroller
c.MPIControllerLauncher.controller_args = ["--ip='*'", "--log-level=20"]
