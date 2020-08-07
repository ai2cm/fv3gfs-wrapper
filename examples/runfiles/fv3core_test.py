import fv3gfs
import fv3core
from fv3util import Quantity
import numpy as np
import xarray as xr
import yaml

# May need to run 'ulimit -s unlimited' before running this example
# If you're running in our prepared docker container, you definitely need to do this
# sets the stack size to unlimited

# Run using mpirun -n 6 python3 basic_model.py
# mpirun flags that may be useful:
#     for docker:  --allow-run-as-root
#     for CircleCI: --oversubscribe
#     to silence a certain inconsequential MPI error: --mca btl_vader_single_copy_mechanism none

# All together:
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 basic_model.py

# def init_tracers(shape):
#     arr = np.zeros(shape)
#     turbulent_kinetic_energy = Quantity.from_data_array(arr)

if __name__ == "__main__":
    
    names0 = ["specific_humidity", "cloud_water_mixing_ratio", "rain_mixing_ratio", "snow_mixing_ratio", "cloud_ice_mixing_ratio", "graupel_mixing_ratio", "ozone_mixing_ratio", "cloud_amount", "air_temperature", "pressure_thickness_of_atmospheric_layer", "vertical_thickness_of_atmospheric_layer", "logarithm_of_interface_pressure", "x_wind", "y_wind", "vertical_wind", "x_wind_on_c_grid", "y_wind_on_c_grid", "total_condensate_mixing_ratio", "interface_pressure", "surface_geopotential", "interface_pressure_raised_to_power_of_kappa", "surface_pressure", "vertical_pressure_velocity", "atmosphere_hybrid_a_coordinate", "atmosphere_hybrid_b_coordinate", "accumulated_x_mass_flux", "accumulated_y_mass_flux", "accumulated_x_courant_number", "accumulated_y_courant_number", "dissipation_estimate_from_heat_source", "ks"]

    names = ["specific_humidity", "cloud_water_mixing_ratio", "rain_mixing_ratio", "snow_mixing_ratio", "cloud_ice_mixing_ratio", "graupel_mixing_ratio", "ozone_mixing_ratio", "cloud_amount", "turbulent_kinetic_energy", "cloud_fraction", "air_temperature", "pressure_thickness_of_atmospheric_layer", "vertical_thickness_of_atmospheric_layer", "logarithm_of_interface_pressure", "x_wind", "y_wind", "vertical_wind", "x_wind_on_a_grid", "y_wind_on_a_grid", "x_wind_on_c_grid", "y_wind_on_c_grid", "total_condensate_mixing_ratio", "interface_pressure", "surface_geopotential", "interface_pressure_raised_to_power_of_kappa", "finite_volume_mean_pressure_raised_to_power_of_kappa", "surface_pressure", "vertical_pressure_velocity", "atmosphere_hybrid_a_coordinate", "atmosphere_hybrid_b_coordinate", "accumulated_x_mass_flux", "accumulated_y_mass_flux", "accumulated_x_courant_number", "accumulated_y_courant_number", "dissipation_estimate_from_heat_source", "ptop", "timestep"]

    # config = yaml.safe_load(open("/fv3gfs-python/examples/runfiles/default.yml", "r"))
    # nsplit = config["namelist"]["fv_core_nml"]["n_split"]
    # consv_te = config["namelist"]["fv_core_nml"]["consv_te"]
    # dt_atmos = config["namelist"]["coupler_nml"]["dt_atmos"]

    fv3gfs.initialize()
    arr = np.zeros((13,13,63))
    turbulent_kinetic_energy = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "qsgs_tke", "units": "m**2/s**2"}))
    cloud_fraction = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "qcld", "units": ""}))
    # air_temperature = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "pt", "units": "degK"}))
    # pressure_thickness_of_atmospheric_layer = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "delp", "units": "Pa"}))
    # vertical_thickness_of_atmospheric_layer = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "delz", "units": "m"}))
    # logarithm_of_interface_pressure = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "peln", "units": "ln(Pa)"}))
    x_wind_on_a_grid = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "ua", "units": "m/s"}))
    y_wind_on_a_grid = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "va", "units": "m/s"}))
    finite_volume_mean_pressure_raised_to_power_of_kappa = Quantity.from_data_array(xr.DataArray(arr, attrs={"fortran_name": "pkz", "units": "unknown"}))
    flags = fv3gfs.Flags
    for i in range(fv3gfs.get_step_count()):
        if i==0:
            state = fv3gfs.get_state(names=names0)
            state["turbulent_kinetic_energy"] = turbulent_kinetic_energy
            state["cloud_fraction"] = cloud_fraction
            # state["air_temperature"] = air_temperature
            # state["pressure_thickness_of_atmospheric_layer"] = pressure_thickness_of_atmospheric_layer
            # state["vertical_thickness_of_atmospheric_layer"] = vertical_thickness_of_atmospheric_layer
            # state["logarithm_of_interface_pressure"] = logarithm_of_interface_pressure
            state["x_wind_on_a_grid"] = x_wind_on_a_grid
            state["y_wind_on_a_grid"] = y_wind_on_a_grid
            state["finite_volume_mean_pressure_raised_to_power_of_kappa"] = finite_volume_mean_pressure_raised_to_power_of_kappa
        else:
            state = fv3gfs.get_state(names=names)
        fv3core.fv_dynamics(state, comm, flags.consv_te, flags.do_adiabatic_init, flags.dt_atmos, flags.ptop, flags.n_split, flags.ks)
        fv3gfs.set_state(state)
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()
    fv3gfs.cleanup()
