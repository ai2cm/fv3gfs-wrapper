ARG WRAPPER_IMAGE
FROM $WRAPPER_IMAGE AS notebook

RUN pip install --no-cache-dir ipyparallel==6.3.0 jupyterlab matplotlib==3.2.2 jedi==0.17.2 && \
    ipython profile create --parallel --profile=mpi
COPY notebooks /notebooks
COPY ipcluster_config.py /root/.ipython/profile_mpi/ipcluster_config.py
WORKDIR /notebooks

CMD ["jupyter", "lab", "--port=8870", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
