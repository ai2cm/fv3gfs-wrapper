import logging
import os
import fsspec
import yaml
import fv3config

logger = logging.getLogger("kubernetes.py")

PARENT_BUCKET = (
    "gs://vcm-ml-data/fv3gfs-python-kubernetes-example/"
)
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = fv3config.get_default_config()
    config_location = os.path.join(PARENT_BUCKET, "fv3config.yml")
    outdir = os.path.join(PARENT_BUCKET, 'outdir')
    fs = fsspec.filesystem('gs')
    with fs.open(config_location, "w") as config_file:
        config_file.write(yaml.dump(config))
        fv3config.run_kubernetes(
            config_location,
            outdir,
            DOCKER_IMAGE,
            jobname='fv3gfs-python-kubernetes-example',
            namespace="default",
            memory_gb=3.6,
            cpu_count=1,
            gcp_secret="gcp-key",
            image_pull_policy="Always",
        )
    logger.info('submitted job')
