.. highlight:: shell


Installation
============

Docker
------

Installation onto your host machine is currently only supported using Docker.
If you would like to run this package on your host directly, please consult
the Dockerfile for the build process and attempt to replicate it as closely
as possible on your machine.

Build
-----

The `fv3gfs-wrapper` docker image can be build with:: console

    $ build_docker.sh

By default this will produce an image ``us.gcr.io/vcm-ml/fv3gfs-wrapper:latest``. You can
rename this image to whatever you like using ``docker tag us.gcr.io/vcm-ml/fv3gfs-wrapper:latest <new_name>``.

Test
----

You can easily build the docker image and then run all tests using:: console

    $ test_docker.sh

If you would like more fine-grained control over the build and test process, you can
run tests directly. Read ``test_docker.sh`` for usage examples.
