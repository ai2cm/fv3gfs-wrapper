#!/bin/bash

set -e

BUILD_FROM_INTERMEDIATE=n BUILD_HPC=y make -C docker
