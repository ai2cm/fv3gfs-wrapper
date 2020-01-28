#-----------------------------------------------------------------------------
FROM ubuntu:19.10 as fv3config-inputdata

RUN apt-get update && \
    apt-get install -y wget && \
    wget http://storage.googleapis.com/vcm-ml-public/2019-10-23-data-for-running-fv3gfs.tar.gz && \
    tar -xf /2019-10-23-data-for-running-fv3gfs.tar.gz --directory /

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

ENV FMS_DIR=/fv3gfs-python/lib/external/FMS
ENV ESMF_DIR=/fv3gfs-python/lib/external/esmf
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

ARG FV3GFS_BUILD_ROOT=/

COPY --from=fv3gfs-fms /FMS $FMS_DIR
COPY --from=fv3gfs-esmf /usr/local/esmf ${ESMF_DIR}

COPY --from=fv3gfs-fms /FMS/libFMS/.libs/*.so* /lib64/
COPY --from=fv3gfs-esmf /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* /lib64/

# copy wrapper and fortran sources
# directory copy commands must be separate due to docker limitations
COPY lib /fv3gfs-python/lib
COPY --from=fv3gfs-fortran-build ${FV3GFS_BUILD_ROOT}/stochastic_physics /fv3gfs-python/lib/external/stochastic_physics
COPY --from=fv3gfs-fortran-build ${FV3GFS_BUILD_ROOT}/FV3 /fv3gfs-python/lib/external/FV3
COPY templates /fv3gfs-python/templates
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

ENV FMS_DIR=/fv3gfs-python/lib/external/FMS
ENV ESMF_DIR=/fv3gfs-python/lib/external/esmf
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

COPY --from=fv3gfs-fms /FMS/libFMS/.libs/*.so* /lib64/
COPY --from=fv3gfs-esmf /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* /lib64/

COPY --from=fv3gfs-python-build /fv3gfs-python /fv3gfs-python
COPY tests /fv3gfs-python/tests

# copy and install fv3config
COPY external /fv3gfs-python/external

RUN pip3 install --no-cache-dir -e /fv3gfs-python && \
    pip3 install --no-cache-dir -e /fv3gfs-python/external/fv3config

ENV FV3CONFIG_CACHE_DIR=/inputdata

# cache model data
COPY --from=fv3config-inputdata /2019-10-23-data-for-running-fv3gfs $FV3CONFIG_CACHE_DIR/fv3config-cache

RUN chmod -R 777 $FV3CONFIG_CACHE_DIR

# make stack size unlimited
RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc

# interactive shell by default
CMD ["bash"]
