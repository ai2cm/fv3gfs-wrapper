SHELL = /bin/sh

ifndef FV3GFS_BUILD_DIR
	FV3GFS_BUILD_DIR=$(shell pwd)/external/FV3
endif
include $(FV3GFS_BUILD_DIR)/conf/configure.fv3

TEMPLATES = $(shell ls ../templates)
PROPERTIES_FILES = ../fv3gfs/dynamics_properties.json ../fv3gfs/physics_properties.json

FFLAGS += $(shell pkg-config fv3 --cflags) \
	-I$(FMS_DIR) \
	-I$(FMS_DIR)/include

.PHONY: clean cleanall fortran_libs

all: coupler_lib.o dynamics_data.o physics_data.o $(TEMPLATES)

dynamics_data.o: dynamics_data.F90
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

physics_data.o: physics_data.F90 dynamics_data.o
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

coupler_lib.o: coupler_lib.F90
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

clean:
	@echo "Cleaning ... "
	$(RM) -f *.o *.mod *.lst *.c depend $(TEMPLATES)
	$(RM) -rf tests/pytest/output/*
	(cd external && make clean)

$(TEMPLATES): %: ../templates/% $(PROPERTIES_FILES)
	python3 ../fill_templates.py $@

cleanall: clean
