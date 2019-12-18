SHELL = /bin/sh

ifndef FV3GFS_BUILD_DIR
	FV3GFS_BUILD_DIR=$(shell pwd)/FV3/sorc/fv3gfs.fd/FV3/
endif
include $(FV3GFS_BUILD_DIR)/conf/configure.fv3

TEMPLATES = $(shell ls ../templates)
PROPERTIES_FILES = ../fv3gfs/dynamics_properties.json ../fv3gfs/physics_properties.json

FFLAGS += -I$(FV3GFS_BUILD_DIR)/fms -I$(FV3GFS_BUILD_DIR)/gfsphysics -I$(FV3GFS_BUILD_DIR)/io -I$(FV3GFS_BUILD_DIR)/atmos_cubed_sphere -I$(FV3GFS_BUILD_DIR)

FORTRAN_LIBS = $(FV3GFS_BUILD_DIR)/atmos_model.o \
	$(FV3GFS_BUILD_DIR)/atmos_cubed_sphere/libfv3core.a \
	$(FV3GFS_BUILD_DIR)/stochastic_physics/libstochastic_physics.a \
	$(FV3GFS_BUILD_DIR)/gfsphysics/libgfsphys.a \
	$(FV3GFS_BUILD_DIR)/io/libfv3io.a \
	$(FV3GFS_BUILD_DIR)/fms/libfms.a

.PHONY: clean cleanall legacy_fortran templates

all: templates coupler_lib.o dynamics_data.o physics_data.o

dynamics_data.o: dynamics_data.F90 $(FORTRAN_LIBS)
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

physics_data.o: physics_data.F90 dynamics_data.o $(FORTRAN_LIBS)
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

coupler_lib.o: coupler_lib.F90 $(FORTRAN_LIBS)
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

clean:
	@echo "Cleaning ... "
	$(RM) -f *.o *.mod *.lst depend $(TEMPLATES)
	$(MAKE) -C $(FV3GFS_BUILD_DIR) clean

templates: $(TEMPLATES)

$(TEMPLATES): %: ../templates/% $(PROPERTIES_FILES)
	python3 ../fill_templates.py $@

cleanall: clean
