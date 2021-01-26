DOCKER_IMAGE=gcr.io/vcm-ml/fv3gfs-wrapper:ams2021
SHELL:=/bin/bash

RUNFILE?=
RUNDIR?=

all: baseline validation random_forest mpi_communication
.PHONY: enter run clean baseline validation random_forest mpi_communication

enter:
	docker run -v $(shell pwd):/workdir -w /workdir -it $(DOCKER_IMAGE) bash

rundir:
	wget https://zenodo.org/record/4429298/files/c48_6h.tar.gz
	tar -xvf c48_6h.tar.gz

run: rundir
	$(RM) -r $(RUNDIR)
	cp -r rundir $(RUNDIR)
	cp $(RUNFILE) $(RUNDIR)/
	cd $(RUNDIR) && time mpirun -n 6 python3 $(RUNFILE)

baseline: rundir
	cp -r rundir baseline
	cd baseline && time mpirun -n 6 fv3.exe

validation: rundir
	RUNFILE=validation.py RUNDIR=validation $(MAKE) run

random_forest: rundir
	RUNFILE=random_forest.py RUNDIR=random_forest $(MAKE) run

mpi_communication: rundir
	RUNFILE=mpi_communication.py RUNDIR=mpi_communication $(MAKE) run

clean:
	$(RM) -rf rundir c48_6h.tar.gz baseline validation random_forest mpi_communication 
