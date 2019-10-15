import unittest
import os
import subprocess
import shutil
import hashlib

base_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
work_dir = os.path.join(base_dir, 'workdir')
fortran_work_dir = os.path.join(base_dir, 'fortran_workdir')
input_dir = '/FV3/inputdata/fv3gfs-data-docker/rundir'  # won't be hard-coded when we get a package for run directories


mpi_flags = [
    "--allow-run-as-root", "--oversubscribe",
    "--mca", "btl_vader_single_copy_mechanism", "none",
]


def get_file_hash(filename):
    hasher = hashlib.md5()
    with open(filename, 'rb') as file:
        hasher.update(file.read())
    return hasher.hexdigest()


class FV3GFSRunTests(unittest.TestCase):

    def setUp(self):
        print('Setting up')
        prepare_workdir(work_dir)
        prepare_workdir(fortran_work_dir)

    def tearDown(self):
        print('Tearing down')
        clear_workdir(work_dir)
        clear_workdir(fortran_work_dir)

    def test_getters(self):
        perform_python_run(os.path.join(base_dir, 'runs/test_getters.py'))

    def test_setters_after_init(self):
        perform_python_run(os.path.join(base_dir, 'runs/test_setters_after_init.py'))

    def test_setters_after_steps(self):
        perform_python_run(os.path.join(base_dir, 'runs/test_setters_after_steps.py'))

    def test_default_run(self):
        perform_python_run(os.path.join(base_dir, 'runs/default_run.py'))
        perform_fortran_run()
        failures = compare_paths(work_dir, fortran_work_dir, exclude_names=['logfile.000000.out'])
        self.assertFalse(failures)


def compare_paths(path1, path2, exclude_names=None):
    """Returns a recursive list of files that are present under both paths, but differ."""
    exclude_names = exclude_names or ()
    path1_filenames = os.listdir(path1)
    path2_filenames = os.listdir(path2)
    shared_names = set(path1_filenames).intersection(path2_filenames)
    valid_shared_names = shared_names.difference(exclude_names)
    failures = []
    for name in valid_shared_names:
        name1 = os.path.join(path1, name)
        name2 = os.path.join(path2, name)
        if os.path.isdir(name1) and os.path.isdir(name2):
            failures.extend(compare_paths(name1, name2))
        elif not files_match_if_present(name1, name2):
            failures.append((name1, name2))
    return failures


def files_match_if_present(name1, name2):
    if not os.path.isfile(name1) or not os.path.isfile(name2):
        return True
    elif get_file_hash(name1) == get_file_hash(name2):
        return True
    else:
        return False


def prepare_workdir(work_dir):
    clear_workdir(work_dir)
    shutil.copytree(input_dir, work_dir)


def clear_workdir(work_dir):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)


def perform_python_run(filename, n_processes=6):
    base_filename = os.path.basename(filename)
    work_filename = os.path.join(work_dir, base_filename)
    shutil.copy2(filename, work_filename)
    subprocess.check_call(
        ["mpirun", "-n", str(n_processes)] + mpi_flags + ["python3", base_filename],
        cwd=work_dir,
    )


def perform_fortran_run(n_processes=6):
    filename = os.path.join(parent_dir, 'fv3.exe')
    base_filename = os.path.basename(filename)
    work_filename = os.path.join(fortran_work_dir, base_filename)
    shutil.copy2(filename, work_filename)
    print(f'Copied {filename} to {work_filename}')
    subprocess.check_call(
        ["mpirun", "-n", str(n_processes)] + mpi_flags + [work_filename],
        cwd=fortran_work_dir,
    )


if __name__ == '__main__':
    unittest.main()
