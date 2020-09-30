SHELL = /bin/sh

TEMPLATES = $(shell ls ../templates)
PROPERTIES_FILES = ../fv3gfs/wrapper/dynamics_properties.json ../fv3gfs/wrapper/physics_properties.json

# Use pkg-config to get flags for fv3
# If we run without PKG_CONFIG_ALLOW_SYSTEM_CFLAGS,
# the -I/usr/include flag will be missing
FFLAGS += $(shell PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config fv3 --cflags)
FC ?= mpif90

.PHONY: clean cleanall fortran_libs

all: coupler_lib.o dynamics_data.o physics_data.o $(TEMPLATES)

%.o: %.F90
	$(FC) $(FFLAGS)  -c $< -o $@

%.o: %.f90
	$(FC) $(FFLAGS)  -c $< -o $@

physics_data.o: physics_data.F90 dynamics_data.o
	$(FC) $(FFLAGS)  -c $< -o $@

coupler_lib.o: coupler_lib.F90
	$(FC) $(FFLAGS)  -c $< -o $@

clean:
	@echo "Cleaning ... "
	$(RM) -f *.o *.mod *.lst *.c depend $(TEMPLATES)
	$(RM) -rf tests/pytest/output/*

$(TEMPLATES): %: ../templates/% $(PROPERTIES_FILES)
	python3 ../fill_templates.py $@

cleanall: clean
