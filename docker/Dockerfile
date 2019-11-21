FROM ubuntu:19.10 AS fv3gfs-env

RUN apt-get update && apt-get install -y \
    nano \
    curl \
    wget \
    gcc \
    git \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    perl \
    make \
    rsync \
    libffi-dev \
    openssl \
    python3 \
    python3-pip \
    libopenmpi3 \
    netcdf-bin \
    libssl-dev \
    python3 \
    python3-numpy \
    python3-mpi4py \
    cython3 \
    python3-pip

# install required python packages
RUN pip3 install --no-cache-dir \
    xarray==0.13.0 \
    jinja2==2.10.3 \
    netcdf4==1.4.2 \
    f90nml==1.1.2 \
    appdirs==1.4.3 \
    requests==2.22.0 \
    Click==7.0 \
    pyyaml==5.1.2

# download and install NCEP libraries
RUN git config --global http.sslverify false && \
    git clone https://github.com/NCAR/NCEPlibs.git && \
    mkdir /opt/NCEPlibs && \
    cd NCEPlibs && \
    git checkout 3da51e139d5cd731c9fc27f39d88cb4e1328212b && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1

#-----------------------------------------------------------------------------
FROM fv3gfs-env AS fv3gfs-python

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
COPY fill_templates.py HISTORY.md LICENSE Makefile MANIFEST.in README.md setup.cfg setup.py /fv3gfs-python/

ARG compile_option
ARG configure_file=configure.fv3.gnu_docker

ENV FV3GFS_BUILD_DIR=/fv3gfs-python/lib/FV3/sorc/fv3gfs.fd/FV3/

# copy appropriate configuration file to configure.fv3
RUN cp $FV3GFS_BUILD_DIR/conf/$configure_file $FV3GFS_BUILD_DIR/conf/configure.fv3 && \
    if [ ! -z $compile_option ]; then sed -i "33i $compile_option" $FV3GFS_BUILD_DIR/conf/configure.fv3; fi

# make sources first, because -j8 doesn't propagate through setup.py layer
RUN cd /fv3gfs-python && \
    make clean && \
    cd /fv3gfs-python/lib && \
    make -j8

# compile wrapper
RUN cd /fv3gfs-python && \
    make build && \
    pip3 install -e /fv3gfs-python

# interactive shell by default
CMD ["bash"]
