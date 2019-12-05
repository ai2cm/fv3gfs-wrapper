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
    python3-mpi4py \
    netcdf-bin \
    libssl-dev \
    python3 \
    python3-numpy \
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
    pyyaml==5.1.2 \
    google-cloud-storage==1.23.0 \
    gcsfs==0.4.0 \
    fsspec==0.6.0 \
    pytest==5.2.2 \
    kubernetes==10.0.1

# download and install NCEP libraries
RUN git config --global http.sslverify false && \
    git clone https://github.com/NCAR/NCEPlibs.git && \
    mkdir /opt/NCEPlibs && \
    cd NCEPlibs && \
    git checkout 3da51e139d5cd731c9fc27f39d88cb4e1328212b && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1

#-----------------------------------------------------------------------------
FROM ubuntu:19.10 as fv3config-inputdata

RUN apt-get update && \
    apt-get install -y wget && \
    wget http://storage.googleapis.com/vcm-ml-public/2019-10-23-data-for-running-fv3gfs.tar.gz && \
    tar -xf /2019-10-23-data-for-running-fv3gfs.tar.gz --directory /

#-----------------------------------------------------------------------------
FROM fv3gfs-env AS fv3gfs-build

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

# copy appropriate configuration file to configure.fv3
RUN cp /fv3gfs-python/lib/FV3/sorc/fv3gfs.fd/FV3/conf/$configure_file \
        /fv3gfs-python/lib/FV3/sorc/fv3gfs.fd/FV3/conf/configure.fv3 && \
    if [ ! -z $compile_option ]; then sed -i "33i $compile_option" \
        /fv3gfs-python/lib/FV3/sorc/fv3gfs.fd/FV3/conf/configure.fv3; fi

# make sources first, because -j8 doesn't propagate through setup.py layer
RUN cd /fv3gfs-python && \
    make clean && \
    cd /fv3gfs-python/lib && \
    make -j8

# compile wrapper
RUN cd /fv3gfs-python && \
    make build

#-----------------------------------------------------------------------------
FROM fv3gfs-env AS fv3gfs-python

COPY --from=fv3gfs-build /fv3gfs-python /fv3gfs-python
RUN pip3 install --no-cache-dir -e /fv3gfs-python

# copy and install fv3config
COPY external /fv3gfs-python/external

RUN pip3 install --no-cache-dir -e /fv3gfs-python/external/fv3config

ENV FV3CONFIG_CACHE_DIR=/inputdata

# cache model data
COPY --from=fv3config-inputdata /2019-10-23-data-for-running-fv3gfs /inputdata/fv3config-cache

RUN chmod -R 755 /inputdata

# interactive shell by default
CMD ["bash"]
