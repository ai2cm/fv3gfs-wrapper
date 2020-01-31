Reference files exist in subdirectories of `reference`, e.g. `reference/circleci` for
the baseline checksums used in Circle CI. This directory is a symbolic link to the same
reference directory in the `fv3gfs-fortran` repo. These tests check regression against
the fortran version of the model.

Regression outputs should not be updated because of changes to this repository. They
should instead be updated as necessary when updating `fv3gfs-fortran`.

Test configurations are stored in `config` as fv3config yaml files. Adding new
yaml files to this directory will add new regression tests automatically. This directory
is also a symbolic link into the `fv3gfs-fortran` repo.
