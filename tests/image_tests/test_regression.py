import os
import yaml
import subprocess
import pytest
import fv3config
import hashlib


TEST_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIR = os.path.join(TEST_DIR, "config")
config_filenames = os.listdir(CONFIG_DIR)


@pytest.fixture(params=config_filenames)
def config(request):
    config_filename = os.path.join(CONFIG_DIR, request.param)
    with open(config_filename, "r") as config_file:
        return yaml.safe_load(config_file)


def md5_from_dir(dir_):
    md5s = {}
    for root, dirs, files in os.walk(str(dir_)):
        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                md5 = hashlib.md5()
                while True:
                    buf = f.read(2048)
                    if not buf:
                        break
                    md5.update(buf)

            md5s[file] = md5.hexdigest()
    return md5s


def md5_from_dir_only_nc(dir_):
    return {
        file: hash for file, hash in md5_from_dir(dir_).items() if file.endswith(".nc")
    }


def test_md5_from_dir(tmpdir):
    tmpdir.join("a").open("w").write("hello")
    tmpdir.join("b").open("w").write("world")

    orig_md5 = md5_from_dir(tmpdir)
    assert orig_md5 == md5_from_dir(tmpdir)

    tmpdir.join("b").open("w").write("world updated")
    assert orig_md5 != md5_from_dir(tmpdir)


def test_regression(tmpdir, config):
    fv3_rundir = tmpdir.join("fv3")
    wrapper_rundir = tmpdir.join("wrapper")

    run_fv3(config, fv3_rundir)
    run_wrapper(config, wrapper_rundir)

    assert md5_from_dir_only_nc(fv3_rundir) == md5_from_dir_only_nc(wrapper_rundir)

    # just make sure there are some outputs
    assert len(md5_from_dir_only_nc(fv3_rundir)) > 0


def run_fv3(config, run_dir):
    fv3config.write_run_directory(config, str(run_dir))
    subprocess.check_call(
        ["mpirun", "-n", "6", "fv3.exe"], cwd=run_dir,
    )


def run_wrapper(config, run_dir):
    fv3config.write_run_directory(config, str(run_dir))
    subprocess.check_call(
        ["mpirun", "-n", "6", "python3", "-m", "mpi4py", "-m", "fv3gfs.wrapper.run"],
        cwd=run_dir,
    )
