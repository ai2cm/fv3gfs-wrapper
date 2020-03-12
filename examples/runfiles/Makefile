
RUNFILE ?=
CONFIG ?= default.yml
OUTDIR ?= outdir
DOCKER_IMAGE ?= us.gcr.io/vcm-ml/fv3gfs-python

CASES=basic_model online_code

all: $(CASES) nudging

public_examples: $(CASES)

fv3run:
	fv3run --runfile $(RUNFILE) --dockerimage $(DOCKER_IMAGE) $(CONFIG) $(OUTDIR)

$(CASES): %:
	RUNFILE=$@.py OUTDIR=$@_outdir $(MAKE) fv3run

nudging:
	RUNFILE=nudging.py OUTDIR=nudging_outdir CONFIG=nudging.yml $(MAKE) fv3run

clean:
	$(RM) -rf outdir nudging_outdir basic_model_outdir online_code_outdir

.PHONY: all fv3run basic_model nudging online_code public_examples
