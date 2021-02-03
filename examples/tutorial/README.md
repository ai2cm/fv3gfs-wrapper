# fv3gfs-wrapper tutorial

This directory contains notebooks used for a tutorial held for the NOAA
Geophysical Fluid Dynamics Laboratory in January of 2021.  The notebooks cover
four topics:

1. Running the base model
2. Using `get_state` to view the state of the running model
3. Using `set_state` to modify the state of the running model
4. Restarting the model through Python

The notebooks contain guided examples as well as exercises. Solutions are
included for the exercises.  You can view them statically in a browser via
GitHub's rendering of Jupyter notebooks, or you can run them interactively on
your own computer following the "quickstart" instructions below.

## Quickstart

Clone the fv3gfs-wrapper repository, navigate to the
`fv3gfs-wrapper/examples/tutorial` directory, and run `make all`.  This will
download the required input dataset [hosted on
Zenodo](https://zenodo.org/record/4429298#.YBq_VC1h1Zh), build a Docker image,
and start a Jupyter Lab server.  Copy and paste the URL provided upon starting
the Jupyter Lab server into a browser; from there, you should be able to work
through the notebooks interactively.

```
$ git clone https://github.com/VulcanClimateModeling/fv3gfs-wrapper.git
$ cd fv3gfs-wrapper/examples/tutorial
$ make all
```
