This directory runs the command-line examples and produces timings as presented in a manuscript to be submitted to Geophysical Model Development (GMD) in 2021.
The Jupyter notebook example presented in that paper is included under examples/jupyter.

To produce the timings, you must have Docker installed and configured with at least 5GB of memory available.

1. First, enter a docker container for fv3gfs-wrapper interactively using `make enter`.
2. From within the docker container, run `make`. This will download the run directory from Zenodo and run the examples within that directory, with timings output at the end of each run.
