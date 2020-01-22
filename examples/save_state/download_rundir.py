import fv3config
import fsspec
import shutil
import yaml

config_url = "gs://vcm-ml-data/2020-01-15-noahb-exploration/2hr_strong_dampingone_step_config/C48/20160805.000000/fv3config.yml"

output_url = ("gs://vcm-ml-data/2020-01-15-noahb-exploration/save_data_test/C48/20160805.000000/")
run_file_path = "save_state_runfile.py"
digest = "sha256:4870c4e91bc5988df222631f27967572bff977be9ddce6195a1260f7192d85a0"

with fsspec.open(config_url) as f:
    config = yaml.load(f)


if __name__ == '__main__':

    fv3config.write_run_directory(config, "rundir")
