import os
from datetime import timedelta
import yaml
import fv3gfs.wrapper
from mpi4py import MPI


REFERENCE_DIR = "gs://vcm-ml-intermediate/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/"
TENDENCY_OUT_FILENAME = "nudging_tendencies.nc"
RUN_DIR = os.path.dirname(os.path.realpath(__file__))


def get_timestep(config):
    return timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])


def get_timescales_from_config(config):
    return_dict = {}
    for name, hours in config["nudging"]["timescale_hours"].items():
        return_dict[name] = timedelta(hours=hours)
    return return_dict


def time_to_label(time):
    return f"{time.year:04d}{time.month:02d}{time.day:02d}.{time.hour:02d}{time.minute:02d}{time.second:02d}"


def get_restart_directory(label):
    return os.path.join(REFERENCE_DIR, label)


def get_reference_state(time, communicator, only_names):
    label = time_to_label(time)
    dirname = get_restart_directory(label)
    return fv3gfs.wrapper.open_restart(
        dirname, communicator, label=label, only_names=only_names
    )


def nudge_to_reference(state, communicator, timescales, timestep):
    reference = get_reference_state(
        state["time"], communicator, only_names=timescales.keys()
    )
    tendencies = fv3gfs.wrapper.apply_nudging(state, reference, timescales, timestep)
    tendencies = append_key_label(tendencies, "_tendency_due_to_nudging")
    tendencies["time"] = state["time"]
    return tendencies


def append_key_label(d, suffix):
    return_dict = {}
    for key, value in d.items():
        return_dict[key + suffix] = value
    return return_dict


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    current_dir = os.getcwd()
    with open("fv3config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    communicator = fv3gfs.wrapper.CubedSphereCommunicator(
        MPI.COMM_WORLD,
        fv3gfs.wrapper.CubedSpherePartitioner.from_namelist(config["namelist"]),
    )
    nudging_timescales = get_timescales_from_config(config)
    timestep = get_timestep(config)
    tendency_monitor = fv3gfs.wrapper.ZarrMonitor(
        os.path.join(RUN_DIR, "tendencies.zarr"),
        communicator.partitioner,
        mode="w",
        mpi_comm=MPI.COMM_WORLD,
    )

    fv3gfs.wrapper.initialize()
    for i in range(fv3gfs.wrapper.get_step_count()):
        fv3gfs.wrapper.step_dynamics()
        fv3gfs.wrapper.step_physics()
        fv3gfs.wrapper.save_intermediate_restart_if_enabled()

        state = fv3gfs.wrapper.get_state(
            names=["time"] + list(nudging_timescales.keys())
        )
        tendencies = nudge_to_reference(
            state, communicator, nudging_timescales, timestep
        )
        tendency_monitor.store(tendencies)
        fv3gfs.wrapper.set_state(state)
    fv3gfs.wrapper.cleanup()
