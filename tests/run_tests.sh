#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

return_value=0

ulimit -s unlimited

python3 $DIR/test_integration.py || return_value=1

#python3 ./test_get_tile_number.py || return_value=1
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 $DIR/test_restart.py || return_value=1
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 $DIR/test_getters.py || return_value=1
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 $DIR/test_setters.py || return_value=1

exit $return_value
