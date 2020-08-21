SHELL = /bin/sh

ifndef FV3GFS_BUILD_DIR
	FV3GFS_BUILD_DIR=$(shell pwd)/external/FV3
endif
include $(FV3GFS_BUILD_DIR)/conf/configure.fv3

TEMPLATES = $(shell ls ../templates)
PROPERTIES_FILES = ../fv3gfs/wrapper/dynamics_properties.json ../fv3gfs/wrapper/physics_properties.json

FFLAGS += \
	-I$(FMS_DIR) \
	-I$(FMS_DIR)/include \
	-I$(FV3GFS_BUILD_DIR)/gfsphysics \
	-I$(FV3GFS_BUILD_DIR)/ipd \
	-I$(FV3GFS_BUILD_DIR)/cpl \
	-I$(FV3GFS_BUILD_DIR)/io \
	-I$(FV3GFS_BUILD_DIR)/atmos_cubed_sphere \
	-I$(FV3GFS_BUILD_DIR)/ccpp/driver \
	-I$(FV3GFS_BUILD_DIR)/../stochastic_physics \
	-I$(FV3GFS_BUILD_DIR)

.PHONY: clean cleanall fortran_libs

all: coupler_lib.o dynamics_data.o physics_data.o $(TEMPLATES)

dynamics_data.o: dynamics_data.F90 fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

physics_data.o: physics_data.F90 dynamics_data.o fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

coupler_lib.o: coupler_lib.F90 fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

clean:
	@echo "Cleaning ... "
	$(RM) -f *.o *.mod *.lst *.c depend $(TEMPLATES)
	$(RM) -rf tests/pytest/output/*
	(cd external && make clean)

$(TEMPLATES): %: ../templates/% $(PROPERTIES_FILES)
	python3 ../fill_templates.py $@

fortran_libs:
	cd $(FV3GFS_BUILD_DIR) && make

cleanall: clean
