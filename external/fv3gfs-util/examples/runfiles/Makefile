
all: global_timings

global_timings:
	PMIX_MCA_gds=hash mpirun -n 4 python -m mpi4py global_timings.py
