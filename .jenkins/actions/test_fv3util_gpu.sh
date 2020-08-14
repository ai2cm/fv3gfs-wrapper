#!/bin/bash -f

set -e

##################################################
# functions
##################################################

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

showUsage()
{
    echo "usage: `basename $0` [-h]"
    echo ""
    echo "optional arguments:"
    echo "-h           show this help message and exit"
}

parseOptions()
{
    # process command line options
    while getopts "h" opt
    do
        case $opt in
        h) showUsage; exit 0 ;;
        \?) showUsage; exitError 301 ${LINENO} "invalid command line option (-${OPTARG})" ;;
        :) showUsage; exitError 302 ${LINENO} "command line option (-${OPTARG}) requires argument" ;;
        esac
    done

}

# echo basic setup
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

# start timer
T="$(date +%s)"

# parse command line options (pass all of them to function)
parseOptions $*

# we only run this on HPC
module load daint-gpu
module load cray-python
module load pycuda

# run tests
echo "### run tests"
if [ ! -f external/fv3util/requirements.txt ] ; then
    exitError 1205 ${LINENO} "could not find requirements.txt, run from top directory"
fi
python3 -m venv venv
module unload cray-python
module unload pycuda
. ./venv/bin/activate
pip3 install -r external/fv3util/requirements.txt
pip3 install external/fv3util cupy-cuda101 pytest==5.4.3
pip3 install git+git://github.com/GridTools/gt4py.git@204838ea0e88a30b6eb41babbd58fb8a86d14a0e
pytest --junitxml results.xml external/fv3util/tests

deactivate

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0

# so long, Earthling!
