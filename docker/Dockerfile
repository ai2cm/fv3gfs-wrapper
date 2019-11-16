ARG FV3GFS_IMAGE=us.gcr.io/vcm-ml/fv3gfs-environment:latest
FROM $FV3GFS_IMAGE

ENV FV3GFS_BUILD_DIR=/fv3gfs-python/lib/FV3/sorc/fv3gfs.fd/FV3/


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

ADD /docker/install_gcloud.sh /setup/install_gcloud.sh
RUN bash /setup/install_gcloud.sh

# install required python packages
RUN pip3 install --no-cache-dir \
    xarray==0.13.0 \
    jinja2==2.10.3 \
    netcdf4==1.4.2 \
    f90nml==1.1.2 \
    appdirs==1.4.3 \
    requests==2.22.0 \
    Click==7.0 \
    pyyaml==5.1.2 \
    google-cloud-storage==1.23.0 \
    gcsfs==0.4.0

# copy and install fv3config first so data cache is only re-downloaded
# if fv3config itself changes
COPY external /fv3gfs-python/external

RUN pip3 install --no-cache-dir -e /fv3gfs-python/external/fv3config

ENV FV3CONFIG_CACHE_DIR=/inputdata

# cache model data
RUN python3 -m fv3config.download_data

# copy wrapper and fortran sources
# directory copy commands must be separate due to docker limitations
COPY fv3gfs /fv3gfs-python/fv3gfs
COPY lib /fv3gfs-python/lib
COPY templates /fv3gfs-python/templates
COPY tests /fv3gfs-python/tests
COPY examples /fv3gfs-python/examples
COPY docs /fv3gfs-python/docs
# files can be copied in one line
COPY fill_templates.py HISTORY.md LICENSE makefile MANIFEST.in README.md setup.cfg setup.py /fv3gfs-python/

ARG compile_option
ARG configure_file=configure.fv3.gnu_docker

# copy appropriate configuration file to configure.fv3
RUN cp $FV3GFS_BUILD_DIR/conf/$configure_file $FV3GFS_BUILD_DIR/conf/configure.fv3 && \
    if [ ! -z $compile_option ]; then sed -i "33i $compile_option" $FV3GFS_BUILD_DIR/conf/configure.fv3; fi

# compile wrapper
RUN cd /fv3gfs-python && \
    make clean && \
    make build -j8 && \
    pip3 install -e /fv3gfs-python


# interactive shell by default
CMD ["bash"]
