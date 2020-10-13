ARG BASE_IMAGE=ubuntu:19.10
ARG MPI_IMAGE
ARG FMS_IMAGE
ARG ESMF_IMAGE
ARG FORTRAN_ENV_IMAGE
ARG GT4PY_CUDA=" "

FROM $MPI_IMAGE AS fv3gfs-mpi
FROM $FMS_IMAGE AS fv3gfs-fms
FROM $ESMF_IMAGE AS fv3gfs-esmf

#-----------------------------------------------------------------------------
FROM $BASE_IMAGE as fv3config-inputdata

RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    pip3 install --no-cache-dir gsutil
RUN mkdir -p /fv3config-cache/gs/
COPY docker/vcm-fv3config /fv3config-cache/gs/vcm-fv3config
RUN    rm -rf /fv3config-cache/gs/vcm-fv3config/data/base_forcing/v1.0 \
        /fv3config-cache/gs/vcm-fv3config/data/orographic_data/v1.0/C384 \
        /fv3config-cache/gs/vcm-fv3config/data/orographic_data/v1.0/C96 \
        /fv3config-cache/orographic_data/C384 \
        /fv3config-cache/base_forcing

#-----------------------------------------------------------------------------
FROM $FORTRAN_ENV_IMAGE as fv3gfs-wrapper-env

COPY requirements.txt /fv3gfs-wrapper/requirements.txt
COPY --from=fv3gfs-mpi /mpich-3.1.4 /mpich-3.1.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-pip \
    cython3 && \
    pip3 install wheel setuptools==50.3.0 && \
    pip3 install -r /fv3gfs-wrapper/requirements.txt  && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

RUN cd /mpich-3.1.4 && \
    make install && ldconfig

#-----------------------------------------------------------------------------
FROM fv3gfs-wrapper-env AS fv3gfs-wrapper-build

ENV FMS_DIR=/fv3gfs-wrapper/lib/external/FMS \
    ESMF_DIR=/fv3gfs-wrapper/lib/external/esmf 

ENV ESMF_INC="-I${ESMF_DIR}/include" \ 
    FMS_LIB=${FMS_DIR}/libFMS/.libs/ \ 
    ESMF_LIB=${ESMF_DIR}/lib  

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

ARG FV3GFS_BUILD_ROOT=/

COPY --from=fv3gfs-fms /FMS $FMS_DIR
COPY --from=fv3gfs-esmf /usr/local/esmf ${ESMF_DIR}
COPY --from=fv3gfs-fms /FMS/libFMS/.libs/*.so* /lib64/
COPY --from=fv3gfs-esmf /usr/local/esmf/lib/*.so* /lib64/

# copy wrapper and fortran sources
# directory copy commands must be separate due to docker limitations
COPY lib /fv3gfs-wrapper/lib
COPY --from=fv3gfs-fortran-build /stochastic_physics/ /fv3gfs-wrapper/lib/external/stochastic_physics/
COPY --from=fv3gfs-fortran-build /FV3/ /fv3gfs-wrapper/lib/external/FV3/

RUN PREFIX=/usr/local make -C /fv3gfs-wrapper/lib/external/FV3 install

COPY templates /fv3gfs-wrapper/templates
COPY examples /fv3gfs-wrapper/examples
COPY docs /fv3gfs-wrapper/docs

# files can be copied in one line
COPY fv3gfs/wrapper/dynamics_properties.json fv3gfs/wrapper/physics_properties.json /fv3gfs-wrapper/fv3gfs/wrapper/
COPY fill_templates.py HISTORY.md LICENSE Makefile MANIFEST.in README.md setup.cfg setup.py /fv3gfs-wrapper/
COPY fv3gfs /fv3gfs-wrapper/fv3gfs

#-----------------------------------------------------------------------------
FROM fv3gfs-wrapper-build AS fv3gfs-wrapper

RUN pip3 install --no-deps --no-cache-dir git+https://github.com/GridTools/gt4py.git@db40777788863bfe87b9674b118e8bd5b8db0440#egg=gt4py$(GT4PY_CUDA) 

# copy and install dependency packages
COPY fill_templates.py /fv3gfs-wrapper/fill_templates.py
COPY templates/ /fv3gfs-wrapper/templates/
COPY external /fv3gfs-wrapper/external
COPY lib/ /fv3gfs-wrapper/lib/

COPY requirements_local.txt /fv3gfs-wrapper/requirements_local.txt
RUN cd /fv3gfs-wrapper && \
    export MPI=mpich && \
    export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH} && \
    make -C lib && \ 
    pip3 install --no-deps --no-cache-dir -r /fv3gfs-wrapper/requirements_local.txt

# cache model data
ENV FV3CONFIG_CACHE_DIR=/inputdata \
    LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH} \
    MPI=mpich
COPY --from=fv3config-inputdata /fv3config-cache $FV3CONFIG_CACHE_DIR/fv3config-cache

RUN chmod -R 777 $FV3CONFIG_CACHE_DIR && \
    echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# copy the tests
COPY tests /fv3gfs-wrapper/tests

# interactive shell by default
CMD ["bash"]
