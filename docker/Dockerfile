ARG FMS_IMAGE
ARG ESMF_IMAGE
ARG FORTRAN_ENV_IMAGE
ARG DAWN_IMAGE=fv3gfs-dawn-build

FROM $FMS_IMAGE AS fv3gfs-fms
FROM $ESMF_IMAGE AS fv3gfs-esmf

#-----------------------------------------------------------------------------
FROM ubuntu:19.10 as fv3config-inputdata

RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    pip3 install --no-cache-dir gsutil
RUN mkdir -p /fv3config-cache/gs/vcm-fv3config && \
    gsutil -m cp -r gs://vcm-fv3config/ /fv3config-cache/gs && \
    rm -rf /fv3config-cache/gs/vcm-fv3config/data/base_forcing/v1.0 \
        /fv3config-cache/gs/vcm-fv3config/data/orographic_data/v1.0/C384 \
        /fv3config-cache/gs/vcm-fv3config/data/orographic_data/v1.0/C96 \
        /fv3config-cache/orographic_data/C384 \
        /fv3config-cache/base_forcing

#-----------------------------------------------------------------------------
FROM $FORTRAN_ENV_IMAGE as fv3gfs-python-env

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-setuptools \
    python3-pytest \
    python3-numpy-dbg \
    python3-pip \
    python3-mpi4py \
    python3-numpy \
    cython3 && \
    pip3 install --no-cache-dir \
    jinja2==2.10.3 \
    xarray==0.13.0 \
    netcdf4==1.4.2 \
    f90nml==1.1.2 \
    requests==2.22.0 \
    pytest==5.4.3 && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

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
COPY metadata /fv3gfs-python/metadata
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
FROM $FORTRAN_ENV_IMAGE AS fv3gfs-dawn-build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cmake \
    libssl-dev \
    libboost-all-dev \ 
    clang \
    clang-format \
    clang-tidy \
    llvm-9-dev \
    libclang-9-dev \
    libpython3-dev \
    python3 \
    python3-pip \
    python3-numpy \
    python3-nose \
    python3-sphinx
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN pip install --no-cache-dir scikit-build && \
    git clone https://github.com/MeteoSwiss-APN/dawn.git /usr/src/dawn && \
    pip install --no-cache-dir /usr/src/dawn/dawn
RUN pip wheel /usr/src/dawn/dawn
RUN git clone --depth 1 -b fv3_validation https://github.com/eddie-c-davis/gt4py.git /usr/src/gt4py && \
    pip install --no-cache-dir -e /usr/src/gt4py && \
    cd /usr/src/gt4py && python -m gt4py.gt_src_manager install


FROM $DAWN_IMAGE AS fv3gfs-dawn

#-----------------------------------------------------------------------------
FROM fv3gfs-python-env AS fv3gfs-python

ENV FMS_DIR=/FMS
ENV ESMF_DIR=/esmf
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

ENV FV3CONFIG_CACHE_DIR=/inputdata

# install fv3config requirements and utility packages
RUN pip3 install --no-cache-dir \
    appdirs==1.4.3 \
    pyyaml==5.1.2 \
    google-cloud-storage==1.23.0 \
    gcsfs==0.6.0 \
    fsspec==0.6.0 \
    kubernetes==10.0.1 \
    numcodecs==0.6.4 \
    zarr==2.3.2 \
    h5py==2.10.0 \
    h5netcdf==0.8.0 \
    pytest-subtests==0.3.1 \
    dask==2.10.1 \
    toolz==0.10.0 \
    scipy==1.3.1 \
    sphinx==2.2.0 \
    sphinx_rtd_theme \
    backoff==1.10 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    nano \
    curl \
    netcdf-bin

COPY --from=fv3gfs-fms /FMS $FMS_DIR
COPY --from=fv3gfs-esmf /usr/local/esmf ${ESMF_DIR}
COPY --from=fv3gfs-fms /FMS/libFMS/.libs/*.so* /lib64/
COPY --from=fv3gfs-esmf /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* /lib64/
COPY --from=fv3gfs-dawn /usr/src/dawn /usr/src/dawn
COPY --from=fv3gfs-dawn /dawn4py-0.0.2-cp37-cp37m-linux_x86_64.whl /usr/src/dawn/dawn4py-0.0.2-cp37-cp37m-linux_x86_64.whl
COPY --from=fv3gfs-dawn /usr/src/gt4py /usr/src/gt4py

RUN pip3 install --no-cache-dir /usr/src/dawn/dawn4py-0.0.2-cp37-cp37m-linux_x86_64.whl
RUN pip3 install --no-cache-dir /usr/src/gt4py

COPY --from=fv3gfs-python-build /fv3gfs-python /fv3gfs-python
COPY tests /fv3gfs-python/tests

# copy and install dependency packages
COPY external /fv3gfs-python/external

RUN pip3 install --no-cache-dir -e \
    /fv3gfs-python/external/fv3config \
    -e /fv3gfs-python/external/fv3util && \
    pip3 install --no-cache-dir -e /fv3gfs-python && \
    pip3 install --no-cache-dir pytest-subtests==0.3.1 pytest==5.4.3  # must reinstall to increment pytest, gt4py requires 4 (but not really)

# cache model data
COPY --from=fv3config-inputdata /fv3config-cache $FV3CONFIG_CACHE_DIR/fv3config-cache

RUN chmod -R 777 $FV3CONFIG_CACHE_DIR && \
    echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# interactive shell by default
CMD ["bash"]
