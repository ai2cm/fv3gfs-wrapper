#!/bin/bash

ulimit -s unlimited
# python3 -m unittest discover --pattern=test_*.py || exit 1
# python3 ./test_get_tile_number.py || exit 1
python3 ./test_aliases.py || exit 1
# python3 ./test_integration.py || exit 1

# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 ./mpi_test_getters.py || exit 1
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 ./mpi_test_setters.py || exit 1
