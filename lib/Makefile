SHELL = /bin/sh
FV3GFS_BUILD_DIR ?= $(shell pwd)/external/FV3

include $(FV3GFS_BUILD_DIR)/conf/configure.fv3

TEMPLATES = $(shell ls ../templates)
PROPERTIES_FILES = ../external/fv3util/fv3util/dynamics_properties.json ../external/fv3util/fv3util/physics_properties.json

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


FV3GFS_RELATIVE_OBJECTS = \
    atmos_model.o \
    module_fv3_config.o \
    cpl/libfv3cpl.a \
    ipd/libipd.a \
    atmos_cubed_sphere/libfv3core.a \
    coarse_graining/libfv3coarse_graining.a \
    io/libfv3io.a \
    gfsphysics/libgfsphys.a \
    ../stochastic_physics/libstochastic_physics.a \

NCEP_LIBS = \
    /opt/NCEPlibs/lib/libnemsio_d.a \
    /opt/NCEPlibs/lib/libbacio_4.a \
    /opt/NCEPlibs/lib/libsp_v2.0.2_d.a \
    /opt/NCEPlibs/lib/libw3emc_d.a \
    /opt/NCEPlibs/lib/libw3nco_d.a \


FV3GFS_OBJECTS = $(addprefix $(FV3GFS_BUILD_DIR)/, $(FV3GFS_RELATIVE_OBJECTS))

LOCAL_OBJECTS = coupler_lib.o

OBJECTS = $(FV3GFS_OBJECTS) $(NCEP_LIBS) $(LOCAL_OBJECTS)


.PHONY: clean cleanall fortran_libs

all: libcoupler.a $(TEMPLATES)

%.o : %.f90 fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

dynamics_data.o: dynamics_data.F90 tracers.o fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

physics_data.o: physics_data.F90 tracers.o dynamics_data.o fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

coupler_lib.o: coupler_lib.F90 fortran_libs
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

libcoupler.a: $(LOCAL_OBJECTS) fortran_libs
	ar crs $@ $(OBJECTS)

clean:
	@echo "Cleaning ... "
	$(RM) *.a -f *.o *.mod *.lst *.c depend $(TEMPLATES)
	$(RM) -rf tests/pytest/output/*
	(cd external && make clean)

$(TEMPLATES): %: ../templates/% $(PROPERTIES_FILES)
	python3 ../fill_templates.py $@

fortran_libs:
	$(MAKE) -C $(FV3GFS_BUILD_DIR) 

cleanall: clean
