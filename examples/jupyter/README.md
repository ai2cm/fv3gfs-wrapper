This directory serves as an example of how you can run fv3gfs-wrapper in a parallel Jupyter notebook using Docker.

Note this proof-of-concept example is not regression tested, and is pinned to a specific version of the wrapper image. Using newer versions of the wrapper as a base image (using WRAPPER_IMAGE) may or may not work. If it does not work with the latest version of the wrapper, please [open an issue](https://github.com/VulcanClimateModeling/fv3gfs-wrapper/issues/new).

This example builds an MPI-enabled Docker image by adding jupyter notebooks to the fv3gfs-wrapper image, and opens it with port forwarding. You can build this image with:

```bash
    $ make build
```

You can open the notebook by running the following, and then going to the URL linked at the bottom of the output:

```bash
    $ make run
```

You should see output which looks something like:

```
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://bf9d4ed4dbc0:8888/?token=f83d5d44389db4e98e888e6862344e051bfb9cf37fbee7fe
     or http://127.0.0.1:8888/?token=f83d5d44389db4e98e888e6862344e051bfb9cf37fbee7fe
```

To use the notebook, you need to copy-paste the second URL with `127.0.0.1` into your browser.
