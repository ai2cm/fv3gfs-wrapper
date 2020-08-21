import logging
import fv3config
import uuid


OUTDIR = "gs://vcm-ml-scratch/fv3gfs-wrapper-kubernetes-example/"
CONFIG_LOCATION = "gs://vcm-fv3config/config/yaml/default/v0.3/fv3config.yml"
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-wrapper"


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    fv3config.run_kubernetes(
        CONFIG_LOCATION,
        OUTDIR,
        DOCKER_IMAGE,
        jobname=f"fv3gfs-wrapper-example-{uuid.uuid4().hex}",
        namespace="default",
        memory_gb=3.6,
        cpu_count=1,
        gcp_secret="gcp-key",
        image_pull_policy="Always",
    )
    logger.info("submitted job")
