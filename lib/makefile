SHELL = /bin/sh

ifndef FV3GFS_BUILD_DIR
$(warning FV3GFS_BUILD_DIR is not set)
else
include $(FV3GFS_BUILD_DIR)/conf/configure.fv3
endif

TEMPLATES = $(shell ls ../templates)

FFLAGS   += -I$(FV3GFS_BUILD_DIR)/fms -I$(FV3GFS_BUILD_DIR)/gfsphysics -I$(FV3GFS_BUILD_DIR)/io -I$(FV3GFS_BUILD_DIR)/atmos_cubed_sphere -I$(FV3GFS_BUILD_DIR)

all: coupler_lib.o dynamics_data.o physics_data.o _wrapper.pyx

%.o: %.F90
	$(FC) $(CPPDEFS) $(CPPFLAGS) $(FPPFLAGS) $(FFLAGS) $(OTHERFLAGS) $(OTHER_FFLAGS) $(ESMF_INC) -c $< -o $@

.PHONY: clean cleanall

clean:
	@echo "Cleaning ... "
	$(RM) -f *.o *.mod *.lst depend $(TEMPLATES)

templates: $(TEMPLATES)

$(TEMPLATES): %: ../templates/%
	python3 ../fill_templates.py $@

cleanall: clean
