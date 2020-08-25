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
FROM $FORTRAN_ENV_IMAGE as fv3gfs-wrapper-env

COPY requirements.txt /fv3gfs-wrapper/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3-dev \
    libopenmpi3 \
    python3-dev \
    python3-setuptools \
    python3-pip \
    cython3 && \
    pip3 install wheel && \
    pip3 install -r /fv3gfs-wrapper/requirements.txt  && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

#-----------------------------------------------------------------------------
FROM fv3gfs-wrapper-env AS fv3gfs-wrapper-build

ENV FMS_DIR=/fv3gfs-wrapper/lib/external/FMS
ENV ESMF_DIR=/fv3gfs-wrapper/lib/external/esmf
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

RUN PREFIX=/usr make -C /fv3gfs-python/lib/external/FV3 install

COPY templates /fv3gfs-python/templates
COPY examples /fv3gfs-python/examples
COPY docs /fv3gfs-python/docs
# files can be copied in one line
COPY fv3gfs/wrapper/dynamics_properties.json fv3gfs/wrapper/physics_properties.json /fv3gfs-wrapper/fv3gfs/wrapper/
COPY fill_templates.py HISTORY.md LICENSE Makefile MANIFEST.in README.md setup.cfg setup.py /fv3gfs-wrapper/
COPY fv3gfs /fv3gfs-wrapper/fv3gfs

RUN make -C /fv3gfs-wrapper build

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
FROM fv3gfs-wrapper-env AS fv3gfs-wrapper

ENV FMS_DIR=/FMS
ENV ESMF_DIR=/esmf
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

ENV FV3CONFIG_CACHE_DIR=/inputdata

COPY --from=fv3gfs-fms /FMS $FMS_DIR
COPY --from=fv3gfs-esmf /usr/local/esmf ${ESMF_DIR}
COPY --from=fv3gfs-fms /FMS/libFMS/.libs/*.so* /lib64/
COPY --from=fv3gfs-esmf /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* /lib64/
COPY --from=fv3gfs-dawn /usr/src/dawn /usr/src/dawn
COPY --from=fv3gfs-dawn /dawn4py-0.0.2-cp37-cp37m-linux_x86_64.whl /usr/src/dawn/dawn4py-0.0.2-cp37-cp37m-linux_x86_64.whl
COPY --from=fv3gfs-dawn /usr/src/gt4py /usr/src/gt4py

RUN pip3 install --no-deps --no-cache-dir /usr/src/dawn/dawn4py-0.0.2-cp37-cp37m-linux_x86_64.whl
RUN pip3 install --no-deps --no-cache-dir /usr/src/gt4py

COPY --from=fv3gfs-wrapper-build /fv3gfs-wrapper /fv3gfs-wrapper
COPY tests /fv3gfs-wrapper/tests

# copy and install dependency packages
COPY external /fv3gfs-wrapper/external
COPY requirements_local.txt /fv3gfs-wrapper/requirements_local.txt
RUN cd /fv3gfs-wrapper && pip3 install --no-deps --no-cache-dir -r /fv3gfs-wrapper/requirements_local.txt

# cache model data
COPY --from=fv3config-inputdata /fv3config-cache $FV3CONFIG_CACHE_DIR/fv3config-cache

RUN chmod -R 777 $FV3CONFIG_CACHE_DIR && \
    echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# interactive shell by default
CMD ["bash"]
