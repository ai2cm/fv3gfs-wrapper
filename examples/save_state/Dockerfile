FROM us.gcr.io/vcm-ml/fv3gfs-python@sha256:4870c4e91bc5988df222631f27967572bff977be9ddce6195a1260f7192d85a0


ADD fv3net-0.1.0-py3-none-any.whl /wheels/fv3net-0.1.0-py3-none-any.whl
RUN pip3 install --no-deps /wheels/fv3net-0.1.0-py3-none-any.whl && \
    pip3 install scikit-learn==0.22 joblib

