
RUNFILE ?=
CONFIG ?= default.yml
OUTDIR ?= outdir
DOCKER_IMAGE ?= us.gcr.io/vcm-ml/fv3gfs-wrapper

CASES=basic_model online_code

all: $(CASES) nudging fortran_state baroclinic

public_examples: $(CASES) fortran_state baroclinic

fv3run:
	fv3run --runfile $(RUNFILE) --dockerimage $(DOCKER_IMAGE) $(CONFIG) $(OUTDIR)

$(CASES): %:
	RUNFILE=$@.py OUTDIR=$@_outdir $(MAKE) fv3run

nudging:
	RUNFILE=nudging.py OUTDIR=nudging_outdir CONFIG=nudging.yml $(MAKE) fv3run

fortran_state:
	RUNFILE=fortran_state.py OUTDIR=fortran_state_ourdir CONFIG=default.yml $(MAKE) fv3run

baroclinic:
	RUNFILE=basic_model.py OUTDIR=baroclinic_outdir CONFIG=baroclinic.yml $(MAKE) fv3run

clean:
	$(RM) -rf outdir nudging_outdir basic_model_outdir online_code_outdir fortran_state_ourdir baroclinic_outdir

.PHONY: all fv3run basic_model nudging online_code public_examples fortran_state baroclinic
