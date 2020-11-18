
RUNFILE ?=
CONFIG ?= default.yml
OUTDIR ?= outdir
DOCKER_IMAGE ?= us.gcr.io/vcm-ml/fv3gfs-wrapper

CASES=basic_model online_code

all: $(CASES) nudging fortran_state baroclinic

public_examples: $(CASES) fortran_state baroclinic

fv3run:
	write_run_directory $(CONFIG) $(OUTDIR)
	cp $(RUNFILE) $(OUTDIR)/$(RUNFILE)
	docker run -v $(GOOGLE_APPLICATION_CREDENTIALS):$(GOOGLE_APPLICATION_CREDENTIALS) \
        --env GOOGLE_APPLICATION_CREDENTIALS=$(GOOGLE_APPLICATION_CREDENTIALS) \
		-w /rundir \
		-v $(shell pwd)/$(OUTDIR):/rundir \
		$(DOCKER_IMAGE) \
		mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 $(RUNFILE)

$(CASES): %:
	RUNFILE=$@.py OUTDIR=$@_outdir $(MAKE) fv3run

nudging:
	RUNFILE=nudging.py OUTDIR=nudging_outdir CONFIG=nudging.yml $(MAKE) fv3run

fortran_state:
	RUNFILE=fortran_state.py OUTDIR=fortran_state_outdir CONFIG=default.yml $(MAKE) fv3run

baroclinic:
	RUNFILE=basic_model.py OUTDIR=baroclinic_outdir CONFIG=baroclinic.yml $(MAKE) fv3run

clean:
	$(RM) -rf outdir nudging_outdir basic_model_outdir online_code_outdir fortran_state_outdir baroclinic_outdir

.PHONY: all fv3run basic_model nudging online_code public_examples fortran_state baroclinic
