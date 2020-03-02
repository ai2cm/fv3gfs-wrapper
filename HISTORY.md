History
=======

latest
------

Major changes:
- In order to facilitate testing of pure python code, this PR splits code that has no dependencies on the Fortran wrapper into its own subpackage, `fv3util`. That code then gets imported and used by the `fv3gfs` python package.
- Removed legacy restart loading code (`load_fortran_restart_folder`). This will be added back in a future PR with better logic for identifying dimensions in restart data, using the properties dictionaries.

Minor changes:
- fixed bug where master branch was not pushing untagged images to GCR
- fixed `fv3util.__version__` variable to indicate the correct version (0.3.0)


0.3.0
-----

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


0.2.1
-----

* Fixed Circle CI tagging logic


0.2.0
-----

* Initial tagged release using public fv3gfs repository


0.1.0
-----

* Initial release.

