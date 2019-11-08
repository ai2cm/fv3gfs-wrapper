ARG FV3GFS_IMAGE=us.gcr.io/vcm-ml/fv3gfs-compiled-default:latest
FROM $FV3GFS_IMAGE

ENV FV3GFS_BUILD_DIR /FV3

# install additional required packages
RUN apt-get update && apt-get install -y \
    nano \
    wget \
    libffi-dev \
    openssl \
    libopenmpi3 \
    libssl-dev \
    python3 \
    python3-numpy \
    python3-mpi4py \
    cython3 \
    python3-pip

# enable python/pip (as opposed to python3/pip3)
RUN ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

# install required python packages
RUN pip3 install --no-cache-dir \
    xarray==0.13.0 \
    jinja2==2.10.3 \
    netcdf4==1.4.2 \
    f90nml==1.1.2 \
    appdirs==1.4.3 \
    requests==2.22.0 \
    Click==7.0

# copy and install fv3config first so data cache does not get re-downloaded when non-fv3config code changes
COPY ./fv3config /fv3config

RUN pip3 install --no-cache-dir -e /fv3config

# cache model data
RUN python3 -m fv3config.download_data

# copy wrapper source files
COPY . /cython_wrapper

# we already copied fv3config to root
RUN rm -rf /cython_wrapper/fv3config

# compile wrapper
RUN cd /cython_wrapper && \
    make clean && \
    make dist && \
    pip3 install -e /cython_wrapper

# interactive shell by default
CMD ["bash"]
