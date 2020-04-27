History
=======

v0.4.1 (2020-04-27)
-------------------

Major changes:
- Updated fv3gfs-fortran submodule to 42f2877a. Important changes are:
  - Addition of `do_gfdl_mp_in_physics` and `do_only_clearsky_rad` namelist parameters.
  - Physical constants in model changed to `GFS_PHYS` versions.
  - Preliminary online coarse-graining code infrastructure introduced.
- fv3config updated to 0.3.2
- Added a Makefile for building docker images in docker/Makefile
- build_docker.sh and CircleCI will by default pull intermediate build dependency images instead of rebuilding them. These can be rebuilt manually using the Makefile in `docker`.
- DummyComm now refuses to accept non-contiguous buffers, the same as mpi4py

Minor changes:
- Replaced latent and sensible heat fortran variable names with instantaneous versions instead of cumulative. Updated units of total_precipitation.
- gcsfs updated to 0.6.0
- added backoff to dockerfile, is a new requirement for fv3config
- create empty outdir in dockerfile, required to allow run_docker to upload to a remote outdir
- add .c files to `make clean` in `lib`. These are produced as build artifacts of cython.

v0.4.0 (2020-04-03)
-------------------

Major changes:
- Added Quantity object to replace DataArray in the state dictionary. Quantity.view[:] provides something akin to DataArray.values, except that it may be a cupy or numpy array. Quantity.values is a read-only numpy array.
- Partitioner has been replaced by TilePartitioner
- Added TilePartitioner and CubedSpherePartitioner objects for domain decomposition.
- Added TileCommunicator and CubedSphereCommunicator objects for inter-rank communication.
- Tile scattering is now in `TileCommunicator.scatter`.
- Added `to_dataset(state)` which turns a state dictionary into an xarray Dataset object.
- Added constants `X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM, Z_INTERFACE_DIM, X_DIMS, Y_DIMS, HORIZONTAL_DIMS`, and `INTERFACE_DIMS`
- updated ZarrMonitor, `get_state`, `set_state`, `open_restart`, `apply_nudging`, `get_nudging_tendencies`, `read_state`, `write_state` and tests to work with a Quantity-based state.
- updated `open_restart` to work with the new communicator objects
- Removed `without_ghost_cells` and `with_ghost_cells`, as a ghost-cell independent view is now available as `Quantity.view`.
- examples are updated for the new code and now run on CircleCI
- gt4py is included in the `fv3gfs-python` image
- `Quantity.from_storage` and `Quantity.storage` are removed - storages are treated the same as if they were numpy/cupy arrays. To retrieve the storage, use `Quantity.data` on a quantity that has been initialized from a storage object.
- All tests using numpy arrays now also run using gt4py `CPUStorage` objects.
- Added `compute_physics` and `apply_physics` functions, which separately do the physics and apply the output of the physics routines to the atmospheric prognostic state.
- Update Fortran sources. Major change is addition of `use_analysis_sst` namelist option.
- getters/setters are added for longitude, latitude, gridcell area, and surface/top of atmosphere upward/downward shortwave/ longwave all-sky/clear-sky fluxes (for combinations which exist in the Fortran model).
- Added a key "fortran_subname" to physics_properties which can currently only be used for 2D physics variables, to indicate the property name for variables stored in arrays of structs. A side-effect is that "fortran_name" is no longer unique for 2D physics variables, since a fortran struct array can contain multiple subname variables under a single fortran_name.
- Tile gather operation is implemented in `TileCommunicator.gather`
- vector halo updates are implemented in `CubedSphereCommunicator.start_vector_halo_update` and `CubedSphereCommunicator.finish_vector_halo_update`.
- Interface-level halo updates were fixed to match fv3gfs behavior. The outermost values on interface level variables are in the compute domain on both ranks bordering it, and are not sent at all during halo update. Instead, the next three interior values are sent.
- Added framework for testing MPI mock against mpi4py, with tests for Scatter, Gather, Send/Recv, and Isend/Irecv.
- ZarrMonitor chunk size along horizontal and vertical dimensions are now as intended, fixing a bug which set the chunk size for the first two dimensions to 1

Minor changes:
- Added C12 regression test for `open_restart`
- added scipy and pytest-subtests to requirements and docker container
- fixed docker container so that `dev_docker.sh` no longer overwrites ESMF and FMS installations
- Removed the old  "integration" tests, as their functionality (at least what was enabled) is covered entirely by the newer "image" tests which are compared against the results from the pure-fortran repo.
- Incremented fv3config commit to include fix to version string
- Add getters/setters for temperature_after_physics, eastward_wind_after_physics, northward_wind_after_physics
- Fixed a bug in `dev_docker.sh` where the Fortran sources weren't being bind-mounted, only the Python files
- tests are added for scalar and vector rotation
- concurrency issues in the MPI mock now raise a custom `ConcurrencyError`.
- Fixed two counter-acting bugs: rotation now rotates in the correct direction, and halo updates now appropriately rotate arrays counter to the relative rotation of the axes on the two tiles, instead of in the same direction as that rotation
- Files are fixed to pass our style checks
- CircleCI tests style checks, will only build image if linting passes (since it's a long job). Pure python tests will run alongside linting tests (since they're fast)
- Makefile at top level only includes configure.fv3 if it exists, since it will not exist during linting tests
- Added tests for `array_chunks` routine used by ZarrMonitor.

v0.3.4 (2020-04-22)
-------------------

Major changes:
- Update fv3gfs-fortran submodule to 42f2877a. Changes physical constants to be "GFS_PHYS" versions. Adds some preliminary online coarse-graining code.

v0.3.3
------

Major changes:
- Update fv3gfs-fortran submodule to 31fc2ee. Adds `do_only_clearsky_rad` namelist option.

v0.3.2
------

Major changes:
- Update fv3gfs-fortran submodule to df28fccc. Most important changes are introduction of `use_analysis_sst` and `do_gfdl_mp_in_physics` namelist options.

v0.3.1
------

Major changes:
- In order to facilitate testing of pure python code, this PR splits code that has no dependencies on the Fortran wrapper into its own subpackage, `fv3util`. That code then gets imported and used by the `fv3gfs` python package.
- Removed legacy restart loading code (`load_fortran_restart_folder`). This will be added back in a future PR with better logic for identifying dimensions in restart data, using the properties dictionaries.
- `Partitioner` class added to handle domain decomposition and inter-rank communication
- filesystem module added to use fsspec to handle remote locations
- Added logic to manage model properties into fv3util. Dynamics and physics properties are stored at the package level in JSON files, while tracer properties are registered externally. This allows support for different microphysics packages and field tables, and allows the wrapper to automatically register the tracers being used by the present model configuration.
- Added `open_restart` routine to load restart files. One rank per tile loads the file, and then distributes the data over MPI to each other process. Can choose to load only a subset of the variables to reduce data transfer when the directory is remote.
- Added `apply_nudging` and `get_nudging_tendencies` functions to nudge a model state to a reference state according to given timescales and a timestep.
- Added `ZarrMonitor` object which stores a sequence of model states in a Zarr store. Data is chunked such that each process is responsible for one chunk (with the last process being responsible for two chunks, in the case of interface-level variables).
- Significant additions and reworks to documentation. Removed outdated build instructions and added docker-centric build instructions. Added developer notes about fv3util.
- Added docs-docker make target to build docs inside docker
- Removed old, unused `get_output_array` routine in the `_wrapper.so`.

Minor changes:
- fixed bug where master branch was not pushing untagged images to GCR
- fixed `fv3util.__version__` variable to indicate the correct version (0.3.0)
- Nudging example runfile added
- Updated target fortran sources, requiring small tweaks to docker build instructions
- Added docstrings to any routines missing them in the API docs
- Removed recommended installation targets for apt-get and reduced the number of RUN commands in docker to reduce disk space requirements for intermediate images
- Added new dependencies for zarr writing and documentation building to docker image
- Changed documentation theme to readthedocs theme
- Added IDE directory to gitignore

v0.3.0
------

Major changes:
- Fortran sources are updated from the older fv3gfs repo to the newer fv3atm repo. Running the model is the same, but small changes to the repo require different namelist options when running the model, so configuration dictionaries will need to be updated accordingly.

Minor changes:
- Fixed name of "patch" version in RELEASE.rst
- Docker build instructions are used directly from the Fortran repo, instead of copying build instructions in two places.
- The new repo requires ESMF and FMS to be built separately
- 32bit builds are disabled, will require fixing the build steps in fv3gfs-fortran to properly pass 32bit build flags when building FMS. Flags can still be sent to fv3atm the same way as before.
- Fortran build sources are back to being a PHONY build target, because make was not able to follow through and build them.
- Regression tests are added that use the regression target data in the fv3gfs-fortran repo. This makes some of our other tests redundant, they may be removed later.
- Configuration dictionaries are explicitly specified for tests instead of using `get_default_config`.
- Commented-out restart tests are re-introduced as fixing restarts on the new repo is a work in progress.

v0.2.1
------

* Fixed Circle CI tagging logic

v0.2.0
------

* Initial tagged release using public fv3gfs repository

v0.1.0
------

* Initial release.
