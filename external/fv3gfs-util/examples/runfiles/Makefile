
all: global_timings zarr_monitor

global_timings:
	PMIX_MCA_gds=hash mpirun -n 4 python -m mpi4py global_timings.py


zarr_monitor:
	mpirun -n 6 python -m mpi4py global_timings.py
