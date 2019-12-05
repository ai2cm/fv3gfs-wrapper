FROM ubuntu:19.10 AS fv3gfs-fortran-env

RUN apt-get update && apt-get install -y \
    gcc \
    git \
    rsync \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    perl \
    make \
    libffi-dev \
    openssl \
    libopenmpi3 \
    libssl-dev

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
FROM fv3gfs-fortran-env AS fv3gfs-fortran-build

COPY lib/FV3 /FV3

ARG compile_option
ARG configure_file=configure.fv3.gnu_docker

# copy appropriate configuration file to configure.fv3
RUN cp /FV3/sorc/fv3gfs.fd/FV3/conf/$configure_file \
        /FV3/sorc/fv3gfs.fd/FV3/conf/configure.fv3 && \
    if [ ! -z $compile_option ]; then sed -i "33i $compile_option" \
        /FV3/sorc/fv3gfs.fd/FV3/conf/configure.fv3; fi

RUN cd /FV3/sorc/fv3gfs.fd/FV3 && make

#-----------------------------------------------------------------------------
FROM fv3gfs-fortran-env as fv3gfs-python-env

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-mpi4py \
    python3-numpy \
    cython3

# install python wrapper requirements
RUN pip3 install --no-cache-dir \
    jinja2==2.10.3 \
    xarray==0.13.0 \
    netcdf4==1.4.2 \
    f90nml==1.1.2 \
    requests==2.22.0 \
    pytest==5.2.2 

#-----------------------------------------------------------------------------
FROM fv3gfs-python-env AS fv3gfs-python-build
# copy wrapper and fortran sources
# directory copy commands must be separate due to docker limitations
COPY lib /fv3gfs-python/lib
COPY --from=fv3gfs-fortran-build /FV3 /fv3gfs-python/lib/FV3
COPY templates /fv3gfs-python/templates
COPY tests /fv3gfs-python/tests
COPY examples /fv3gfs-python/examples
COPY docs /fv3gfs-python/docs
# files can be copied in one line
COPY fv3gfs/dynamics_properties.json fv3gfs/physics_properties.json /fv3gfs-python/fv3gfs/
COPY fill_templates.py HISTORY.md LICENSE Makefile MANIFEST.in README.md setup.cfg setup.py /fv3gfs-python/

# make sources first, because -j8 doesn't propagate through setup.py layer
RUN cd /fv3gfs-python/lib && \
    make -j8

COPY fv3gfs /fv3gfs-python/fv3gfs

# compile wrapper
RUN cd /fv3gfs-python && \
    make build

#-----------------------------------------------------------------------------
FROM fv3gfs-python-env AS fv3gfs-python

# install fv3config requirements and utility packages
RUN pip3 install --no-cache-dir \
    appdirs==1.4.3 \
    pyyaml==5.1.2 \
    google-cloud-storage==1.23.0 \
    gcsfs==0.4.0 \
    fsspec==0.6.0 \
    kubernetes==10.0.1 && \
    apt-get update && \
    apt-get install -y \
    wget \
    nano \
    curl \
    netcdf-bin

COPY --from=fv3gfs-python-build /fv3gfs-python /fv3gfs-python

# copy and install fv3config
COPY external /fv3gfs-python/external

RUN pip3 install --no-cache-dir -e /fv3gfs-python && \
    pip3 install --no-cache-dir -e /fv3gfs-python/external/fv3config

ENV FV3CONFIG_CACHE_DIR=/inputdata

# cache model data
COPY --from=fv3config-inputdata /2019-10-23-data-for-running-fv3gfs /inputdata/fv3config-cache

RUN chmod -R 755 /inputdata

# interactive shell by default
CMD ["bash"]
