#!/usr/bin/env nix-shell
#! nix-shell -i bash /fv3gfs-wrapper/shell.nix

export PYTHONPATH=/fv3gfs-wrapper:$PYTHONPATH

exec $@
