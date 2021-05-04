Test configurations are stored in `config` as fv3config yaml files. Adding new
yaml files to this directory will add new regression tests automatically. This directory
is also a symbolic link into the `fv3gfs-fortran` repo.

These tests use pytest-regtest to manage regression data. Some changes are
expected to break bit-for-bit compatibility. Examples include updating a
compiler version or changing compiler flags. In these cases, the checksums
can be updated by running

    pytest --regtest-reset <path to this dir>/test_regression.py
