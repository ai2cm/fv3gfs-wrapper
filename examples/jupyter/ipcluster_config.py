# Configuration file for ipcluster.

# ------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
# ------------------------------------------------------------------------------
## This is an application.

## The date format used by logging formatters for %(asctime)s
#  Default: '%Y-%m-%d %H:%M:%S'
# c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  Default: '[%(name)s]%(highlevel)s %(message)s'
# c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
#  Default: 30
# c.Application.log_level = 30

## Instead of starting the Application, dump configuration to stdout
#  Default: False
# c.Application.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  Default: False
# c.Application.show_config_json = False

# ------------------------------------------------------------------------------
# BaseIPythonApplication(Application) configuration
# ------------------------------------------------------------------------------
## IPython: an enhanced interactive Python shell.

## Whether to create profile dir if it doesn't exist
#  Default: False
# c.BaseIPythonApplication.auto_create = False

## Whether to install the default config files into the profile dir. If a new
#  profile is being created, and IPython contains config files for that profile,
#  then they will be staged into the new directory.  Otherwise, default config
#  files will be automatically generated.
#  Default: False
# c.BaseIPythonApplication.copy_config_files = False

## Path to an extra config file to load.
#
#  If specified, load this config file in addition to any other IPython config.
#  Default: ''
# c.BaseIPythonApplication.extra_config_file = ''

## The name of the IPython directory. This directory is used for logging
#  configuration (through profiles), history storage, etc. The default is usually
#  $HOME/.ipython. This option can also be specified through the environment
#  variable IPYTHONDIR.
#  Default: ''
# c.BaseIPythonApplication.ipython_dir = ''

## The date format used by logging formatters for %(asctime)s
#  See also: Application.log_datefmt
# c.BaseIPythonApplication.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  See also: Application.log_format
# c.BaseIPythonApplication.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  See also: Application.log_level
# c.BaseIPythonApplication.log_level = 30

## Whether to overwrite existing config files when copying
#  Default: False
# c.BaseIPythonApplication.overwrite = False

## The IPython profile to use.
#  Default: 'default'
# c.BaseIPythonApplication.profile = 'default'

## Instead of starting the Application, dump configuration to stdout
#  See also: Application.show_config
# c.BaseIPythonApplication.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  See also: Application.show_config_json
# c.BaseIPythonApplication.show_config_json = False

## Create a massive crash report when IPython encounters what may be an internal
#  error.  The default is to append a short message to the usual traceback
#  Default: False
# c.BaseIPythonApplication.verbose_crash = False

# ------------------------------------------------------------------------------
# BaseParallelApplication(BaseIPythonApplication) configuration
# ------------------------------------------------------------------------------
## IPython: an enhanced interactive Python shell.

## Whether to create profile dir if it doesn't exist
#  See also: BaseIPythonApplication.auto_create
# c.BaseParallelApplication.auto_create = False

## whether to cleanup old logfiles before starting
#  Default: False
# c.BaseParallelApplication.clean_logs = False

## String id to add to runtime files, to prevent name collisions when using
#  multiple clusters with a single profile simultaneously.
#
#  When set, files will be named like: 'ipcontroller-<cluster_id>-engine.json'
#
#  Since this is text inserted into filenames, typical recommendations apply:
#  Simple character strings are ideal, and spaces are not recommended (but should
#  generally work).
#  Default: ''
# c.BaseParallelApplication.cluster_id = ''

## Whether to install the default config files into the profile dir.
#  See also: BaseIPythonApplication.copy_config_files
# c.BaseParallelApplication.copy_config_files = False

## Path to an extra config file to load.
#  See also: BaseIPythonApplication.extra_config_file
# c.BaseParallelApplication.extra_config_file = ''

##
#  See also: BaseIPythonApplication.ipython_dir
# c.BaseParallelApplication.ipython_dir = ''

## The date format used by logging formatters for %(asctime)s
#  See also: Application.log_datefmt
# c.BaseParallelApplication.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  See also: Application.log_format
# c.BaseParallelApplication.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  See also: Application.log_level
# c.BaseParallelApplication.log_level = 30

## whether to log to a file
#  Default: False
# c.BaseParallelApplication.log_to_file = False
c.BaseParallelApplication.log_to_file = True

## The ZMQ URL of the iplogger to aggregate logging.
#  Default: ''
# c.BaseParallelApplication.log_url = ''

## Whether to overwrite existing config files when copying
#  See also: BaseIPythonApplication.overwrite
# c.BaseParallelApplication.overwrite = False

## The IPython profile to use.
#  See also: BaseIPythonApplication.profile
# c.BaseParallelApplication.profile = 'default'

## Instead of starting the Application, dump configuration to stdout
#  See also: Application.show_config
# c.BaseParallelApplication.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  See also: Application.show_config_json
# c.BaseParallelApplication.show_config_json = False

## Create a massive crash report when IPython encounters what may be an
#  See also: BaseIPythonApplication.verbose_crash
# c.BaseParallelApplication.verbose_crash = False

## Set the working dir for the process.
#  Default: '/home/user/.ipython'
# c.BaseParallelApplication.work_dir = '/home/user/.ipython'

# ------------------------------------------------------------------------------
# IPClusterEngines(BaseParallelApplication) configuration
# ------------------------------------------------------------------------------
## Whether to create profile dir if it doesn't exist
#  See also: BaseIPythonApplication.auto_create
# c.IPClusterEngines.auto_create = False

## whether to cleanup old logfiles before starting
#  See also: BaseParallelApplication.clean_logs
# c.IPClusterEngines.clean_logs = False

## String id to add to runtime files, to prevent name collisions when
#  See also: BaseParallelApplication.cluster_id
# c.IPClusterEngines.cluster_id = ''

## Whether to install the default config files into the profile dir.
#  See also: BaseIPythonApplication.copy_config_files
# c.IPClusterEngines.copy_config_files = False

## Daemonize the ipcluster program. This implies --log-to-file. Not available on
#  Windows.
#  Default: False
# c.IPClusterEngines.daemonize = False

## The timeout (in seconds)
#  Default: 30
# c.IPClusterEngines.early_shutdown = 30

## Deprecated, use engine_launcher_class
#  Default: None
# c.IPClusterEngines.engine_launcher = None

## The class for launching a set of Engines. Change this value to use various
#  batch systems to launch your engines, such as PBS,SGE,MPI,etc. Each launcher
#  class has its own set of configuration options, for making sure it will work
#  in your environment.
#
#  You can also write your own launcher, and specify it's absolute import path,
#  as in 'mymodule.launcher.FTLEnginesLauncher`.
#
#  IPython's bundled examples include:
#
#      Local : start engines locally as subprocesses [default]
#      MPI : use mpiexec to launch engines in an MPI environment
#      PBS : use PBS (qsub) to submit engines to a batch queue
#      SGE : use SGE (qsub) to submit engines to a batch queue
#      LSF : use LSF (bsub) to submit engines to a batch queue
#      SSH : use SSH to start the controller
#                  Note that SSH does *not* move the connection files
#                  around, so you will likely have to do this manually
#                  unless the machines are on a shared file system.
#      HTCondor : use HTCondor to submit engines to a batch queue
#      Slurm : use Slurm to submit engines to a batch queue
#      WindowsHPC : use Windows HPC
#
#  If you are using one of IPython's builtin launchers, you can specify just the
#  prefix, e.g:
#
#      c.IPClusterEngines.engine_launcher_class = 'SSH'
#
#  or:
#
#      ipcluster start --engines=MPI
#  Default: 'LocalEngineSetLauncher'
c.IPClusterEngines.engine_launcher_class = "MPI"

## Path to an extra config file to load.
#  See also: BaseIPythonApplication.extra_config_file
# c.IPClusterEngines.extra_config_file = ''

##
#  See also: BaseIPythonApplication.ipython_dir
# c.IPClusterEngines.ipython_dir = ''

## The date format used by logging formatters for %(asctime)s
#  See also: Application.log_datefmt
# c.IPClusterEngines.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  See also: Application.log_format
# c.IPClusterEngines.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  See also: Application.log_level
# c.IPClusterEngines.log_level = 30

## whether to log to a file
#  See also: BaseParallelApplication.log_to_file
# c.IPClusterEngines.log_to_file = False

## The ZMQ URL of the iplogger to aggregate logging.
#  See also: BaseParallelApplication.log_url
# c.IPClusterEngines.log_url = ''

## The number of engines to start. The default is to use one for each CPU on your
#  machine
#  Default: 40
# c.IPClusterEngines.n = 40

## Whether to overwrite existing config files when copying
#  See also: BaseIPythonApplication.overwrite
# c.IPClusterEngines.overwrite = False

## The IPython profile to use.
#  See also: BaseIPythonApplication.profile
# c.IPClusterEngines.profile = 'default'

## Instead of starting the Application, dump configuration to stdout
#  See also: Application.show_config
# c.IPClusterEngines.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  See also: Application.show_config_json
# c.IPClusterEngines.show_config_json = False

## Create a massive crash report when IPython encounters what may be an
#  See also: BaseIPythonApplication.verbose_crash
# c.IPClusterEngines.verbose_crash = False

## Set the working dir for the process.
#  See also: BaseParallelApplication.work_dir
# c.IPClusterEngines.work_dir = '/home/user/.ipython'

# ------------------------------------------------------------------------------
# IPClusterStart(IPClusterEngines) configuration
# ------------------------------------------------------------------------------
## whether to create the profile_dir if it doesn't exist
#  Default: True
# c.IPClusterStart.auto_create = True

## whether to cleanup old logs before starting
#  Default: True
# c.IPClusterStart.clean_logs = True

## String id to add to runtime files, to prevent name collisions when
#  See also: BaseParallelApplication.cluster_id
# c.IPClusterStart.cluster_id = ''

## Set the IP address of the controller.
#  Default: ''
# c.IPClusterStart.controller_ip = ''

## Deprecated, use controller_launcher_class
#  Default: None
# c.IPClusterStart.controller_launcher = None

## The class for launching a Controller. Change this value if you want your
#  controller to also be launched by a batch system, such as PBS,SGE,MPI,etc.
#
#  Each launcher class has its own set of configuration options, for making sure
#  it will work in your environment.
#
#  Note that using a batch launcher for the controller *does not* put it in the
#  same batch job as the engines, so they will still start separately.
#
#  IPython's bundled examples include:
#
#      Local : start engines locally as subprocesses
#      MPI : use mpiexec to launch the controller in an MPI universe
#      PBS : use PBS (qsub) to submit the controller to a batch queue
#      SGE : use SGE (qsub) to submit the controller to a batch queue
#      LSF : use LSF (bsub) to submit the controller to a batch queue
#      HTCondor : use HTCondor to submit the controller to a batch queue
#      Slurm : use Slurm to submit engines to a batch queue
#      SSH : use SSH to start the controller
#      WindowsHPC : use Windows HPC
#
#  If you are using one of IPython's builtin launchers, you can specify just the
#  prefix, e.g:
#
#      c.IPClusterStart.controller_launcher_class = 'SSH'
#
#  or:
#
#      ipcluster start --controller=MPI
#  Default: 'LocalControllerLauncher'
# c.IPClusterStart.controller_launcher_class = 'LocalControllerLauncher'

## Set the location (hostname or ip) of the controller.
#
#  This is used by engines and clients to locate the controller when the
#  controller listens on all interfaces
#  Default: ''
# c.IPClusterStart.controller_location = ''

## Whether to install the default config files into the profile dir.
#  See also: BaseIPythonApplication.copy_config_files
# c.IPClusterStart.copy_config_files = False

## Daemonize the ipcluster program. This implies --log-to-file.
#  See also: IPClusterEngines.daemonize
# c.IPClusterStart.daemonize = False

## delay (in s) between starting the controller and the engines
#  Default: 1.0
# c.IPClusterStart.delay = 1.0

## The timeout (in seconds)
#  See also: IPClusterEngines.early_shutdown
# c.IPClusterStart.early_shutdown = 30

## Deprecated, use engine_launcher_class
#  See also: IPClusterEngines.engine_launcher
# c.IPClusterStart.engine_launcher = None

## The class for launching a set of Engines. Change this value
#  See also: IPClusterEngines.engine_launcher_class
# c.IPClusterStart.engine_launcher_class = 'LocalEngineSetLauncher'

## Path to an extra config file to load.
#  See also: BaseIPythonApplication.extra_config_file
# c.IPClusterStart.extra_config_file = ''

##
#  See also: BaseIPythonApplication.ipython_dir
# c.IPClusterStart.ipython_dir = ''

## The date format used by logging formatters for %(asctime)s
#  See also: Application.log_datefmt
# c.IPClusterStart.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  See also: Application.log_format
# c.IPClusterStart.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  See also: Application.log_level
# c.IPClusterStart.log_level = 30

## whether to log to a file
#  See also: BaseParallelApplication.log_to_file
# c.IPClusterStart.log_to_file = False

## The ZMQ URL of the iplogger to aggregate logging.
#  See also: BaseParallelApplication.log_url
# c.IPClusterStart.log_url = ''

## The number of engines to start. The default is to use one for each
#  See also: IPClusterEngines.n
# c.IPClusterStart.n = 40

## Whether to overwrite existing config files when copying
#  See also: BaseIPythonApplication.overwrite
# c.IPClusterStart.overwrite = False

## The IPython profile to use.
#  See also: BaseIPythonApplication.profile
# c.IPClusterStart.profile = 'default'

## Whether to reset config files as part of '--create'.
#  Default: False
# c.IPClusterStart.reset = False

## Instead of starting the Application, dump configuration to stdout
#  See also: Application.show_config
# c.IPClusterStart.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  See also: Application.show_config_json
# c.IPClusterStart.show_config_json = False

## Create a massive crash report when IPython encounters what may be an
#  See also: BaseIPythonApplication.verbose_crash
# c.IPClusterStart.verbose_crash = False

## Set the working dir for the process.
#  See also: BaseParallelApplication.work_dir
# c.IPClusterStart.work_dir = '/home/user/.ipython'

# ------------------------------------------------------------------------------
# ProfileDir(LoggingConfigurable) configuration
# ------------------------------------------------------------------------------
## An object to manage the profile directory and its resources.
#
#  The profile directory is used by all IPython applications, to manage
#  configuration, logging and security.
#
#  This object knows how to find, create and manage these directories. This
#  should be used by any code that wants to handle profiles.

## Set the profile location directly. This overrides the logic used by the
#  `profile` option.
#  Default: ''
# c.ProfileDir.location = ''

# ------------------------------------------------------------------------------
# LocalEngineSetLauncher(LocalEngineLauncher) configuration
# ------------------------------------------------------------------------------
## Launch a set of engines as regular external processes.

## delay (in seconds) between starting each engine after the first. This can help
#  force the engines to get their ids in order, or limit process flood when
#  starting many engines.
#  Default: 0.1
# c.LocalEngineSetLauncher.delay = 0.1

## command-line arguments to pass to ipengine
#  Default: ['--log-level=20']
# c.LocalEngineSetLauncher.engine_args = ['--log-level=20']

## command to launch the Engine.
#  Default: ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']
# c.LocalEngineSetLauncher.engine_cmd = ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']

# ------------------------------------------------------------------------------
# MPILauncher(LocalProcessLauncher) configuration
# ------------------------------------------------------------------------------
## Launch an external process using mpiexec.

## The command line arguments to pass to mpiexec.
#  Default: []
c.MPILauncher.mpi_args = []

## The mpiexec command to use in starting the process.
#  Default: ['mpiexec']
c.MPILauncher.mpi_cmd = ["mpiexec"]

# ------------------------------------------------------------------------------
# MPIControllerLauncher(MPILauncher) configuration
# ------------------------------------------------------------------------------
## Launch a controller using mpiexec.

## command-line args to pass to ipcontroller
#  Default: ['--log-level=20']
c.MPIControllerLauncher.controller_args = ["--ip='*'", "--log-level=20"]

## Popen command to launch ipcontroller.
#  Default: ['/home/user/venv/bin/python', '-m', 'ipyparallel.controller']
# c.MPIControllerLauncher.controller_cmd = ['/home/user/venv/bin/python', '-m', 'ipyparallel.controller']

## The command line arguments to pass to mpiexec.
#  See also: MPILauncher.mpi_args
# c.MPIControllerLauncher.mpi_args = []

## The mpiexec command to use in starting the process.
#  See also: MPILauncher.mpi_cmd
# c.MPIControllerLauncher.mpi_cmd = ['mpiexec']

# ------------------------------------------------------------------------------
# MPIEngineSetLauncher(MPILauncher) configuration
# ------------------------------------------------------------------------------
## Launch engines using mpiexec

## command-line arguments to pass to ipengine
#  Default: ['--log-level=20']
# c.MPIEngineSetLauncher.engine_args = ['--log-level=20']

## command to launch the Engine.
#  Default: ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']
# c.MPIEngineSetLauncher.engine_cmd = ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']

## The command line arguments to pass to mpiexec.
#  See also: MPILauncher.mpi_args
# c.MPIEngineSetLauncher.mpi_args = []

## The mpiexec command to use in starting the process.
#  See also: MPILauncher.mpi_cmd
# c.MPIEngineSetLauncher.mpi_cmd = ['mpiexec']

# ------------------------------------------------------------------------------
# SSHLauncher(LocalProcessLauncher) configuration
# ------------------------------------------------------------------------------
## A minimal launcher for ssh.
#
#  To be useful this will probably have to be extended to use the ``sshx`` idea
#  for environment variables.  There could be other things this needs as well.

## hostname on which to launch the program
#  Default: ''
# c.SSHLauncher.hostname = ''

## user@hostname location for ssh in one setting
#  Default: ''
# c.SSHLauncher.location = ''

## args to pass to scp
#  Default: []
# c.SSHLauncher.scp_args = []

## command for sending files
#  Default: ['scp']
# c.SSHLauncher.scp_cmd = ['scp']

## args to pass to ssh
#  Default: ['-tt']
# c.SSHLauncher.ssh_args = ['-tt']

## command for starting ssh
#  Default: ['ssh']
# c.SSHLauncher.ssh_cmd = ['ssh']

## List of (remote, local) files to fetch after starting
#  Default: []
# c.SSHLauncher.to_fetch = []

## List of (local, remote) files to send before starting
#  Default: []
# c.SSHLauncher.to_send = []

## username for ssh
#  Default: ''
# c.SSHLauncher.user = ''

# ------------------------------------------------------------------------------
# SSHClusterLauncher(SSHLauncher) configuration
# ------------------------------------------------------------------------------
## hostname on which to launch the program
#  See also: SSHLauncher.hostname
# c.SSHClusterLauncher.hostname = ''

## user@hostname location for ssh in one setting
#  See also: SSHLauncher.location
# c.SSHClusterLauncher.location = ''

## The remote profile_dir to use.
#
#  If not specified, use calling profile, stripping out possible leading homedir.
#  Default: ''
# c.SSHClusterLauncher.remote_profile_dir = ''

## args to pass to scp
#  See also: SSHLauncher.scp_args
# c.SSHClusterLauncher.scp_args = []

## command for sending files
#  See also: SSHLauncher.scp_cmd
# c.SSHClusterLauncher.scp_cmd = ['scp']

## args to pass to ssh
#  See also: SSHLauncher.ssh_args
# c.SSHClusterLauncher.ssh_args = ['-tt']

## command for starting ssh
#  See also: SSHLauncher.ssh_cmd
# c.SSHClusterLauncher.ssh_cmd = ['ssh']

## List of (remote, local) files to fetch after starting
#  See also: SSHLauncher.to_fetch
# c.SSHClusterLauncher.to_fetch = []

## List of (local, remote) files to send before starting
#  See also: SSHLauncher.to_send
# c.SSHClusterLauncher.to_send = []

## username for ssh
#  See also: SSHLauncher.user
# c.SSHClusterLauncher.user = ''

# ------------------------------------------------------------------------------
# SSHControllerLauncher(SSHClusterLauncher) configuration
# ------------------------------------------------------------------------------
## command-line args to pass to ipcontroller
#  Default: ['--log-level=20']
# c.SSHControllerLauncher.controller_args = ['--log-level=20']

## Popen command to launch ipcontroller.
#  Default: ['/home/user/venv/bin/python', '-m', 'ipyparallel.controller']
# c.SSHControllerLauncher.controller_cmd = ['/home/user/venv/bin/python', '-m', 'ipyparallel.controller']

## hostname on which to launch the program
#  See also: SSHLauncher.hostname
# c.SSHControllerLauncher.hostname = ''

## user@hostname location for ssh in one setting
#  See also: SSHLauncher.location
# c.SSHControllerLauncher.location = ''

## The remote profile_dir to use.
#  See also: SSHClusterLauncher.remote_profile_dir
# c.SSHControllerLauncher.remote_profile_dir = ''

## args to pass to scp
#  See also: SSHLauncher.scp_args
# c.SSHControllerLauncher.scp_args = []

## command for sending files
#  See also: SSHLauncher.scp_cmd
# c.SSHControllerLauncher.scp_cmd = ['scp']

## args to pass to ssh
#  See also: SSHLauncher.ssh_args
# c.SSHControllerLauncher.ssh_args = ['-tt']

## command for starting ssh
#  See also: SSHLauncher.ssh_cmd
# c.SSHControllerLauncher.ssh_cmd = ['ssh']

## List of (remote, local) files to fetch after starting
#  See also: SSHLauncher.to_fetch
# c.SSHControllerLauncher.to_fetch = []

## List of (local, remote) files to send before starting
#  See also: SSHLauncher.to_send
# c.SSHControllerLauncher.to_send = []

## username for ssh
#  See also: SSHLauncher.user
# c.SSHControllerLauncher.user = ''

# ------------------------------------------------------------------------------
# SSHEngineLauncher(SSHClusterLauncher) configuration
# ------------------------------------------------------------------------------
## command-line arguments to pass to ipengine
#  Default: ['--log-level=20']
# c.SSHEngineLauncher.engine_args = ['--log-level=20']

## command to launch the Engine.
#  Default: ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']
# c.SSHEngineLauncher.engine_cmd = ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']

## hostname on which to launch the program
#  See also: SSHLauncher.hostname
# c.SSHEngineLauncher.hostname = ''

## user@hostname location for ssh in one setting
#  See also: SSHLauncher.location
# c.SSHEngineLauncher.location = ''

## The remote profile_dir to use.
#  See also: SSHClusterLauncher.remote_profile_dir
# c.SSHEngineLauncher.remote_profile_dir = ''

## args to pass to scp
#  See also: SSHLauncher.scp_args
# c.SSHEngineLauncher.scp_args = []

## command for sending files
#  See also: SSHLauncher.scp_cmd
# c.SSHEngineLauncher.scp_cmd = ['scp']

## args to pass to ssh
#  See also: SSHLauncher.ssh_args
# c.SSHEngineLauncher.ssh_args = ['-tt']

## command for starting ssh
#  See also: SSHLauncher.ssh_cmd
# c.SSHEngineLauncher.ssh_cmd = ['ssh']

## List of (remote, local) files to fetch after starting
#  See also: SSHLauncher.to_fetch
# c.SSHEngineLauncher.to_fetch = []

## List of (local, remote) files to send before starting
#  See also: SSHLauncher.to_send
# c.SSHEngineLauncher.to_send = []

## username for ssh
#  See also: SSHLauncher.user
# c.SSHEngineLauncher.user = ''

# ------------------------------------------------------------------------------
# SSHEngineSetLauncher(LocalEngineSetLauncher) configuration
# ------------------------------------------------------------------------------
## delay (in seconds) between starting each engine after the first.
#  See also: LocalEngineSetLauncher.delay
# c.SSHEngineSetLauncher.delay = 0.1

## command-line arguments to pass to ipengine
#  Default: ['--log-level=20']
# c.SSHEngineSetLauncher.engine_args = ['--log-level=20']

## command to launch the Engine.
#  Default: ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']
# c.SSHEngineSetLauncher.engine_cmd = ['/home/user/venv/bin/python', '-m', 'ipyparallel.engine']

## dict of engines to launch.  This is a dict by hostname of ints, corresponding
#  to the number of engines to start on that host.
#  Default: {}
# c.SSHEngineSetLauncher.engines = {}

# ------------------------------------------------------------------------------
# SSHProxyEngineSetLauncher(SSHClusterLauncher) configuration
# ------------------------------------------------------------------------------
## Launcher for calling `ipcluster engines` on a remote machine.
#
#  Requires that remote profile is already configured.

## hostname on which to launch the program
#  See also: SSHLauncher.hostname
# c.SSHProxyEngineSetLauncher.hostname = ''

#  Default: ['ipcluster']
# c.SSHProxyEngineSetLauncher.ipcluster_cmd = ['ipcluster']

## user@hostname location for ssh in one setting
#  See also: SSHLauncher.location
# c.SSHProxyEngineSetLauncher.location = ''

## The remote profile_dir to use.
#  See also: SSHClusterLauncher.remote_profile_dir
# c.SSHProxyEngineSetLauncher.remote_profile_dir = ''

## args to pass to scp
#  See also: SSHLauncher.scp_args
# c.SSHProxyEngineSetLauncher.scp_args = []

## command for sending files
#  See also: SSHLauncher.scp_cmd
# c.SSHProxyEngineSetLauncher.scp_cmd = ['scp']

## args to pass to ssh
#  See also: SSHLauncher.ssh_args
# c.SSHProxyEngineSetLauncher.ssh_args = ['-tt']

## command for starting ssh
#  See also: SSHLauncher.ssh_cmd
# c.SSHProxyEngineSetLauncher.ssh_cmd = ['ssh']

## List of (remote, local) files to fetch after starting
#  See also: SSHLauncher.to_fetch
# c.SSHProxyEngineSetLauncher.to_fetch = []

## List of (local, remote) files to send before starting
#  See also: SSHLauncher.to_send
# c.SSHProxyEngineSetLauncher.to_send = []

## username for ssh
#  See also: SSHLauncher.user
# c.SSHProxyEngineSetLauncher.user = ''

# ------------------------------------------------------------------------------
# WindowsHPCLauncher(BaseLauncher) configuration
# ------------------------------------------------------------------------------
## The command for submitting jobs.
#  Default: 'job'
# c.WindowsHPCLauncher.job_cmd = 'job'

## The filename of the instantiated job script.
#  Default: 'ipython_job.xml'
# c.WindowsHPCLauncher.job_file_name = 'ipython_job.xml'

## A regular expression used to get the job id from the output of the
#  submit_command.
#  Default: '\\d+'
# c.WindowsHPCLauncher.job_id_regexp = '\\d+'

## The hostname of the scheduler to submit the job to.
#  Default: ''
# c.WindowsHPCLauncher.scheduler = ''

# ------------------------------------------------------------------------------
# WindowsHPCControllerLauncher(WindowsHPCLauncher) configuration
# ------------------------------------------------------------------------------
## The command for submitting jobs.
#  See also: WindowsHPCLauncher.job_cmd
# c.WindowsHPCControllerLauncher.job_cmd = 'job'

## WinHPC xml job file.
#  Default: 'ipcontroller_job.xml'
# c.WindowsHPCControllerLauncher.job_file_name = 'ipcontroller_job.xml'

## A regular expression used to get the job id from the output of the
#  See also: WindowsHPCLauncher.job_id_regexp
# c.WindowsHPCControllerLauncher.job_id_regexp = '\\d+'

## The hostname of the scheduler to submit the job to.
#  See also: WindowsHPCLauncher.scheduler
# c.WindowsHPCControllerLauncher.scheduler = ''

# ------------------------------------------------------------------------------
# WindowsHPCEngineSetLauncher(WindowsHPCLauncher) configuration
# ------------------------------------------------------------------------------
## The command for submitting jobs.
#  See also: WindowsHPCLauncher.job_cmd
# c.WindowsHPCEngineSetLauncher.job_cmd = 'job'

## jobfile for ipengines job
#  Default: 'ipengineset_job.xml'
# c.WindowsHPCEngineSetLauncher.job_file_name = 'ipengineset_job.xml'

## A regular expression used to get the job id from the output of the
#  See also: WindowsHPCLauncher.job_id_regexp
# c.WindowsHPCEngineSetLauncher.job_id_regexp = '\\d+'

## The hostname of the scheduler to submit the job to.
#  See also: WindowsHPCLauncher.scheduler
# c.WindowsHPCEngineSetLauncher.scheduler = ''

# ------------------------------------------------------------------------------
# BatchSystemLauncher(BaseLauncher) configuration
# ------------------------------------------------------------------------------
## Launch an external process using a batch system.
#
#  This class is designed to work with UNIX batch systems like PBS, LSF,
#  GridEngine, etc.  The overall model is that there are different commands like
#  qsub, qdel, etc. that handle the starting and stopping of the process.
#
#  This class also has the notion of a batch script. The ``batch_template``
#  attribute can be set to a string that is a template for the batch script. This
#  template is instantiated using string formatting. Thus the template can use
#  {n} fot the number of instances. Subclasses can add additional variables to
#  the template dict.

## The filename of the instantiated batch script.
#  Default: 'batch_script'
# c.BatchSystemLauncher.batch_file_name = 'batch_script'

## The string that is the batch script template itself.
#  Default: ''
# c.BatchSystemLauncher.batch_template = ''

## The file that contains the batch template.
#  Default: ''
# c.BatchSystemLauncher.batch_template_file = ''

## The name of the command line program used to delete jobs.
#  Default: ['']
# c.BatchSystemLauncher.delete_command = ['']

## A regular expression used to get the job id from the output of the
#  submit_command.
#  Default: ''
# c.BatchSystemLauncher.job_id_regexp = ''

## The group we wish to match in job_id_regexp (0 to match all)
#  Default: 0
# c.BatchSystemLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#
#  This lets you parameterize additional options, such as wall_time with a custom
#  template.
#  Default: {}
# c.BatchSystemLauncher.namespace = {}

## The batch queue.
#  Default: ''
# c.BatchSystemLauncher.queue = ''

## The name of the command line program used to submit jobs.
#  Default: ['']
# c.BatchSystemLauncher.submit_command = ['']

# ------------------------------------------------------------------------------
# PBSLauncher(BatchSystemLauncher) configuration
# ------------------------------------------------------------------------------
## A BatchSystemLauncher subclass for PBS.

## The filename of the instantiated batch script.
#  See also: BatchSystemLauncher.batch_file_name
# c.PBSLauncher.batch_file_name = 'batch_script'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.PBSLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.PBSLauncher.batch_template_file = ''

## The PBS delete command ['qdel']
#  Default: ['qdel']
# c.PBSLauncher.delete_command = ['qdel']

## Regular expresion for identifying the job ID [r'\d+']
#  Default: '\\d+'
# c.PBSLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.PBSLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.PBSLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.PBSLauncher.queue = ''

## The PBS submit command ['qsub']
#  Default: ['qsub']
# c.PBSLauncher.submit_command = ['qsub']

# ------------------------------------------------------------------------------
# PBSControllerLauncher(PBSLauncher) configuration
# ------------------------------------------------------------------------------
## Launch a controller using PBS.

## batch file name for the controller job.
#  Default: 'pbs_controller'
# c.PBSControllerLauncher.batch_file_name = 'pbs_controller'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.PBSControllerLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.PBSControllerLauncher.batch_template_file = ''

## The PBS delete command ['qdel']
#  See also: PBSLauncher.delete_command
# c.PBSControllerLauncher.delete_command = ['qdel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: PBSLauncher.job_id_regexp
# c.PBSControllerLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.PBSControllerLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.PBSControllerLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.PBSControllerLauncher.queue = ''

## The PBS submit command ['qsub']
#  See also: PBSLauncher.submit_command
# c.PBSControllerLauncher.submit_command = ['qsub']

# ------------------------------------------------------------------------------
# PBSEngineSetLauncher(PBSLauncher) configuration
# ------------------------------------------------------------------------------
## Launch Engines using PBS

## batch file name for the engine(s) job.
#  Default: 'pbs_engines'
# c.PBSEngineSetLauncher.batch_file_name = 'pbs_engines'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.PBSEngineSetLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.PBSEngineSetLauncher.batch_template_file = ''

## The PBS delete command ['qdel']
#  See also: PBSLauncher.delete_command
# c.PBSEngineSetLauncher.delete_command = ['qdel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: PBSLauncher.job_id_regexp
# c.PBSEngineSetLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.PBSEngineSetLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.PBSEngineSetLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.PBSEngineSetLauncher.queue = ''

## The PBS submit command ['qsub']
#  See also: PBSLauncher.submit_command
# c.PBSEngineSetLauncher.submit_command = ['qsub']

# ------------------------------------------------------------------------------
# SlurmLauncher(BatchSystemLauncher) configuration
# ------------------------------------------------------------------------------
## A BatchSystemLauncher subclass for slurm.

## Slurm account to be used
#  Default: ''
# c.SlurmLauncher.account = ''

## The filename of the instantiated batch script.
#  See also: BatchSystemLauncher.batch_file_name
# c.SlurmLauncher.batch_file_name = 'batch_script'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.SlurmLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.SlurmLauncher.batch_template_file = ''

## The slurm delete command ['scancel']
#  Default: ['scancel']
# c.SlurmLauncher.delete_command = ['scancel']

## Regular expresion for identifying the job ID [r'\d+']
#  Default: '\\d+'
# c.SlurmLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.SlurmLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.SlurmLauncher.namespace = {}

## Extra Slurm options
#  Default: ''
# c.SlurmLauncher.options = ''

## Slurm QoS to be used
#  Default: ''
# c.SlurmLauncher.qos = ''

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.SlurmLauncher.queue = ''

## The slurm submit command ['sbatch']
#  Default: ['sbatch']
# c.SlurmLauncher.submit_command = ['sbatch']

## Slurm timelimit to be used
#  Default: ''
# c.SlurmLauncher.timelimit = ''

# ------------------------------------------------------------------------------
# SlurmControllerLauncher(SlurmLauncher) configuration
# ------------------------------------------------------------------------------
## Launch a controller using Slurm.

## Slurm account to be used
#  See also: SlurmLauncher.account
# c.SlurmControllerLauncher.account = ''

## batch file name for the controller job.
#  Default: 'slurm_controller.sbatch'
# c.SlurmControllerLauncher.batch_file_name = 'slurm_controller.sbatch'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.SlurmControllerLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.SlurmControllerLauncher.batch_template_file = ''

## The slurm delete command ['scancel']
#  See also: SlurmLauncher.delete_command
# c.SlurmControllerLauncher.delete_command = ['scancel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: SlurmLauncher.job_id_regexp
# c.SlurmControllerLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.SlurmControllerLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.SlurmControllerLauncher.namespace = {}

## Extra Slurm options
#  See also: SlurmLauncher.options
# c.SlurmControllerLauncher.options = ''

## Slurm QoS to be used
#  See also: SlurmLauncher.qos
# c.SlurmControllerLauncher.qos = ''

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.SlurmControllerLauncher.queue = ''

## The slurm submit command ['sbatch']
#  See also: SlurmLauncher.submit_command
# c.SlurmControllerLauncher.submit_command = ['sbatch']

## Slurm timelimit to be used
#  See also: SlurmLauncher.timelimit
# c.SlurmControllerLauncher.timelimit = ''

# ------------------------------------------------------------------------------
# SlurmEngineSetLauncher(SlurmLauncher) configuration
# ------------------------------------------------------------------------------
## Launch Engines using Slurm

## Slurm account to be used
#  See also: SlurmLauncher.account
# c.SlurmEngineSetLauncher.account = ''

## batch file name for the engine(s) job.
#  Default: 'slurm_engine.sbatch'
# c.SlurmEngineSetLauncher.batch_file_name = 'slurm_engine.sbatch'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.SlurmEngineSetLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.SlurmEngineSetLauncher.batch_template_file = ''

## The slurm delete command ['scancel']
#  See also: SlurmLauncher.delete_command
# c.SlurmEngineSetLauncher.delete_command = ['scancel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: SlurmLauncher.job_id_regexp
# c.SlurmEngineSetLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.SlurmEngineSetLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.SlurmEngineSetLauncher.namespace = {}

## Extra Slurm options
#  See also: SlurmLauncher.options
# c.SlurmEngineSetLauncher.options = ''

## Slurm QoS to be used
#  See also: SlurmLauncher.qos
# c.SlurmEngineSetLauncher.qos = ''

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.SlurmEngineSetLauncher.queue = ''

## The slurm submit command ['sbatch']
#  See also: SlurmLauncher.submit_command
# c.SlurmEngineSetLauncher.submit_command = ['sbatch']

## Slurm timelimit to be used
#  See also: SlurmLauncher.timelimit
# c.SlurmEngineSetLauncher.timelimit = ''

# ------------------------------------------------------------------------------
# SGELauncher(PBSLauncher) configuration
# ------------------------------------------------------------------------------
## Sun GridEngine is a PBS clone with slightly different syntax

## The filename of the instantiated batch script.
#  See also: BatchSystemLauncher.batch_file_name
# c.SGELauncher.batch_file_name = 'batch_script'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.SGELauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.SGELauncher.batch_template_file = ''

## The PBS delete command ['qdel']
#  See also: PBSLauncher.delete_command
# c.SGELauncher.delete_command = ['qdel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: PBSLauncher.job_id_regexp
# c.SGELauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.SGELauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.SGELauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.SGELauncher.queue = ''

## The PBS submit command ['qsub']
#  See also: PBSLauncher.submit_command
# c.SGELauncher.submit_command = ['qsub']

# ------------------------------------------------------------------------------
# SGEControllerLauncher(SGELauncher) configuration
# ------------------------------------------------------------------------------
## Launch a controller using SGE.

## batch file name for the ipontroller job.
#  Default: 'sge_controller'
# c.SGEControllerLauncher.batch_file_name = 'sge_controller'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.SGEControllerLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.SGEControllerLauncher.batch_template_file = ''

## The PBS delete command ['qdel']
#  See also: PBSLauncher.delete_command
# c.SGEControllerLauncher.delete_command = ['qdel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: PBSLauncher.job_id_regexp
# c.SGEControllerLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.SGEControllerLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.SGEControllerLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.SGEControllerLauncher.queue = ''

## The PBS submit command ['qsub']
#  See also: PBSLauncher.submit_command
# c.SGEControllerLauncher.submit_command = ['qsub']

# ------------------------------------------------------------------------------
# SGEEngineSetLauncher(SGELauncher) configuration
# ------------------------------------------------------------------------------
## Launch Engines with SGE

## batch file name for the engine(s) job.
#  Default: 'sge_engines'
# c.SGEEngineSetLauncher.batch_file_name = 'sge_engines'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.SGEEngineSetLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.SGEEngineSetLauncher.batch_template_file = ''

## The PBS delete command ['qdel']
#  See also: PBSLauncher.delete_command
# c.SGEEngineSetLauncher.delete_command = ['qdel']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: PBSLauncher.job_id_regexp
# c.SGEEngineSetLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.SGEEngineSetLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.SGEEngineSetLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.SGEEngineSetLauncher.queue = ''

## The PBS submit command ['qsub']
#  See also: PBSLauncher.submit_command
# c.SGEEngineSetLauncher.submit_command = ['qsub']

# ------------------------------------------------------------------------------
# LSFLauncher(BatchSystemLauncher) configuration
# ------------------------------------------------------------------------------
## A BatchSystemLauncher subclass for LSF.

## The filename of the instantiated batch script.
#  See also: BatchSystemLauncher.batch_file_name
# c.LSFLauncher.batch_file_name = 'batch_script'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.LSFLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.LSFLauncher.batch_template_file = ''

## The PBS delete command ['bkill']
#  Default: ['bkill']
# c.LSFLauncher.delete_command = ['bkill']

## Regular expresion for identifying the job ID [r'\d+']
#  Default: '\\d+'
# c.LSFLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.LSFLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.LSFLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.LSFLauncher.queue = ''

## The PBS submit command ['bsub']
#  Default: ['bsub']
# c.LSFLauncher.submit_command = ['bsub']

# ------------------------------------------------------------------------------
# LSFControllerLauncher(LSFLauncher) configuration
# ------------------------------------------------------------------------------
## Launch a controller using LSF.

## batch file name for the controller job.
#  Default: 'lsf_controller'
# c.LSFControllerLauncher.batch_file_name = 'lsf_controller'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.LSFControllerLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.LSFControllerLauncher.batch_template_file = ''

## The PBS delete command ['bkill']
#  See also: LSFLauncher.delete_command
# c.LSFControllerLauncher.delete_command = ['bkill']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: LSFLauncher.job_id_regexp
# c.LSFControllerLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.LSFControllerLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.LSFControllerLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.LSFControllerLauncher.queue = ''

## The PBS submit command ['bsub']
#  See also: LSFLauncher.submit_command
# c.LSFControllerLauncher.submit_command = ['bsub']

# ------------------------------------------------------------------------------
# LSFEngineSetLauncher(LSFLauncher) configuration
# ------------------------------------------------------------------------------
## Launch Engines using LSF

## batch file name for the engine(s) job.
#  Default: 'lsf_engines'
# c.LSFEngineSetLauncher.batch_file_name = 'lsf_engines'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.LSFEngineSetLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.LSFEngineSetLauncher.batch_template_file = ''

## The PBS delete command ['bkill']
#  See also: LSFLauncher.delete_command
# c.LSFEngineSetLauncher.delete_command = ['bkill']

## Regular expresion for identifying the job ID [r'\d+']
#  See also: LSFLauncher.job_id_regexp
# c.LSFEngineSetLauncher.job_id_regexp = '\\d+'

## The group we wish to match in job_id_regexp (0 to match all)
#  See also: BatchSystemLauncher.job_id_regexp_group
# c.LSFEngineSetLauncher.job_id_regexp_group = 0

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.LSFEngineSetLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.LSFEngineSetLauncher.queue = ''

## The PBS submit command ['bsub']
#  See also: LSFLauncher.submit_command
# c.LSFEngineSetLauncher.submit_command = ['bsub']

# ------------------------------------------------------------------------------
# HTCondorLauncher(BatchSystemLauncher) configuration
# ------------------------------------------------------------------------------
## A BatchSystemLauncher subclass for HTCondor.
#
#  HTCondor requires that we launch the ipengine/ipcontroller scripts rather that
#  the python instance but otherwise is very similar to PBS.  This is because
#  HTCondor destroys sys.executable when launching remote processes - a launched
#  python process depends on sys.executable to effectively evaluate its module
#  search paths. Without it, regardless of which python interpreter you launch
#  you will get the to built in module search paths.
#
#  We use the ip{cluster, engine, controller} scripts as our executable to
#  circumvent this - the mechanism of shebanged scripts means that the python
#  binary will be launched with argv[0] set to the *location of the ip{cluster,
#  engine, controller} scripts on the remote node*. This means you need to take
#  care that:
#
#  a. Your remote nodes have their paths configured correctly, with the ipengine and ipcontroller
#     of the python environment you wish to execute code in having top precedence.
#  b. This functionality is untested on Windows.
#
#  If you need different behavior, consider making you own template.

## The filename of the instantiated batch script.
#  See also: BatchSystemLauncher.batch_file_name
# c.HTCondorLauncher.batch_file_name = 'batch_script'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.HTCondorLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.HTCondorLauncher.batch_template_file = ''

## The HTCondor delete command ['condor_rm']
#  Default: ['condor_rm']
# c.HTCondorLauncher.delete_command = ['condor_rm']

## Regular expression for identifying the job ID [r'(\d+)\.$']
#  Default: '(\\d+)\\.$'
# c.HTCondorLauncher.job_id_regexp = '(\\d+)\\.$'

## The group we wish to match in job_id_regexp [1]
#  Default: 1
# c.HTCondorLauncher.job_id_regexp_group = 1

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.HTCondorLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.HTCondorLauncher.queue = ''

## The HTCondor submit command ['condor_submit']
#  Default: ['condor_submit']
# c.HTCondorLauncher.submit_command = ['condor_submit']

# ------------------------------------------------------------------------------
# HTCondorControllerLauncher(HTCondorLauncher) configuration
# ------------------------------------------------------------------------------
## Launch a controller using HTCondor.

## batch file name for the controller job.
#  Default: 'htcondor_controller'
# c.HTCondorControllerLauncher.batch_file_name = 'htcondor_controller'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.HTCondorControllerLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.HTCondorControllerLauncher.batch_template_file = ''

## The HTCondor delete command ['condor_rm']
#  See also: HTCondorLauncher.delete_command
# c.HTCondorControllerLauncher.delete_command = ['condor_rm']

## Regular expression for identifying the job ID [r'(\d+)\.$']
#  See also: HTCondorLauncher.job_id_regexp
# c.HTCondorControllerLauncher.job_id_regexp = '(\\d+)\\.$'

## The group we wish to match in job_id_regexp [1]
#  See also: HTCondorLauncher.job_id_regexp_group
# c.HTCondorControllerLauncher.job_id_regexp_group = 1

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.HTCondorControllerLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.HTCondorControllerLauncher.queue = ''

## The HTCondor submit command ['condor_submit']
#  See also: HTCondorLauncher.submit_command
# c.HTCondorControllerLauncher.submit_command = ['condor_submit']

# ------------------------------------------------------------------------------
# HTCondorEngineSetLauncher(HTCondorLauncher) configuration
# ------------------------------------------------------------------------------
## Launch Engines using HTCondor

## batch file name for the engine(s) job.
#  Default: 'htcondor_engines'
# c.HTCondorEngineSetLauncher.batch_file_name = 'htcondor_engines'

## The string that is the batch script template itself.
#  See also: BatchSystemLauncher.batch_template
# c.HTCondorEngineSetLauncher.batch_template = ''

## The file that contains the batch template.
#  See also: BatchSystemLauncher.batch_template_file
# c.HTCondorEngineSetLauncher.batch_template_file = ''

## The HTCondor delete command ['condor_rm']
#  See also: HTCondorLauncher.delete_command
# c.HTCondorEngineSetLauncher.delete_command = ['condor_rm']

## Regular expression for identifying the job ID [r'(\d+)\.$']
#  See also: HTCondorLauncher.job_id_regexp
# c.HTCondorEngineSetLauncher.job_id_regexp = '(\\d+)\\.$'

## The group we wish to match in job_id_regexp [1]
#  See also: HTCondorLauncher.job_id_regexp_group
# c.HTCondorEngineSetLauncher.job_id_regexp_group = 1

## Extra variables to pass to the template.
#  See also: BatchSystemLauncher.namespace
# c.HTCondorEngineSetLauncher.namespace = {}

## The batch queue.
#  See also: BatchSystemLauncher.queue
# c.HTCondorEngineSetLauncher.queue = ''

## The HTCondor submit command ['condor_submit']
#  See also: HTCondorLauncher.submit_command
# c.HTCondorEngineSetLauncher.submit_command = ['condor_submit']
