#!/bin/bash

set -e

BUILD_FROM_INTERMEDIATE=n HPC_BUILD=gnu make -C docker 
