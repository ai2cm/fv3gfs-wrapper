import os
import datetime as dt
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from argparse import ArgumentParser

usage = "usage: python %(prog)s <output directory> [optional 2nd output directory] [other options]"
parser = ArgumentParser(usage=usage)

parser.add_argument("dir1", type=str, action='store', help="directory containing outputs to plot")
parser.add_argument("dir2", type=str, action='store', help="directory containing outputs to compare to", nargs='?')
args=parser.parse_args()

jw_colors = ['#ec1b8c', '#a6228e', '#20419a', '#0085cc', '#03aeef', '#03aa4f', '#c8da2c', '#fff200', '#f99e1c', '#ed1c24']

np.set_printoptions(precision=14)

####################
## Data Wrangling ##
####################

files = [r'outstate_0.nc', r'outstate_1.nc', r'outstate_2.nc', r'outstate_3.nc', r'outstate_4.nc', r'outstate_5.nc']

ps_plots = []
stemp_plots = []

# Variables:
# ['interface_pressure_raised_to_power_of_kappa',
#  'atmosphere_hybrid_a_coordinate',
#  'atmosphere_hybrid_b_coordinate',
#  'x_wind',
#  'accumulated_y_mass_flux',
#  'y_wind_on_c_grid',
#  'y_wind',
#  'accumulated_x_mass_flux',
#  'x_wind_on_c_grid',
#  'accumulated_x_courant_number',
#  'accumulated_y_courant_number',
#  'eastward_wind',
#  'northward_wind',
#  'air_temperature',
#  'pressure_thickness_of_atmospheric_layer',
#  'vertical_wind',
#  'vertical_pressure_velocity',
#  'vertical_thickness_of_atmospheric_layer',
#  'total_condensate_mixing_ratio',
#  'layer_mean_pressure_raised_to_power_of_kappa',
#  'dissipation_estimate_from_heat_source',
#  'specific_humidity',
#  'cloud_water_mixing_ratio',
#  'rain_mixing_ratio',
#  'cloud_ice_mixing_ratio',
#  'snow_mixing_ratio',
#  'graupel_mixing_ratio',
#  'ozone_mixing_ratio',
#  'turbulent_kinetic_energy',
#  'cloud_fraction',
#  'interface_pressure',
#  'logarithm_of_interface_pressure',
#  'surface_geopotential',
#  'surface_pressure',
#  'time']

vardict = {}

for f in files:
    fname = args.dir1 + f
    ncfile = Dataset(fname, 'r')
    nc_attrs = ncfile.ncattrs()
    nc_dims = [dim for dim in ncfile.dimensions]  # list of nc dimensions
    nc_vars = [var for var in ncfile.variables]  # list of nc variables

    ps = ncfile.variables['surface_pressure'][:].data/100. #convert to hPa
    temp = ncfile.variables['air_temperature'][:].data
    # pl = ncfile.variables['plev'][:]

    surf_temp = temp[-1,:,:] #temp at bottom

    if args.dir2:
        fname2 = args.dir2 + f
        ncf2 = Dataset(fname2, 'r')
        ps2 = ncf2.variables['surface_pressure'][:].data/100. #convert to hPa
        temp2 = ncf2.variables['air_temperature'][:].data
        # pl2 = ncf2.variables['plev'][:]
        surf_temp2 = temp2[-1,:,:] #field at 850 hPa

        ps_diff = (ps - ps2)/ps2
        temp_diff = (surf_temp - surf_temp2)/surf_temp2

        ps_plots.append(ps_diff)
        stemp_plots.append(temp_diff)

        # savin' variables
        for var in nc_vars:
            if "time" not in var:
                if var not in vardict.keys():
                    vardict[var] = []
                field1 = ncfile[var][:]
                field2 = ncf2[var][:]
                vardict[var].append((field1-field2)/field2)


    else:
        ps_plots.append(ps)
        stemp_plots.append(surf_temp)


######################
## Doin' some stats ##
######################
if args.dir2:
    pminmaxes = []
    pmeans = []
    pl1 = []
    pl2 = []
    plinf = []

    tpminmaxes = []
    tpmeans = []
    tpl1 = []
    tpl2 = []
    tplinf = []

    vminmaxes = []
    vmeans = []
    vl1 = []
    vl2 = []
    vlinf = []

    p=np.array(ps_plots)
    tp=np.array(stemp_plots)
    pres_date = np.concatenate((p[0,:,:],p[1,:,:],p[2,:,:],p[3,:,:],p[4,:,:],p[5,:,:]),axis=0)
    pminmaxes.append([pres_date.min(),pres_date.max()])
    pmeans.append(pres_date.mean())
    pl1.append(np.linalg.norm(pres_date.flatten(), 1))
    pl2.append(np.linalg.norm(pres_date.flatten(), 2))
    plinf.append(np.linalg.norm(pres_date.flatten(), np.inf))

    stemp_date = np.concatenate((tp[0,:,:],tp[1,:,:],tp[2,:,:],tp[3,:,:],tp[4,:,:],tp[5,:,:]),axis=0)
    tpminmaxes.append([stemp_date.min(),stemp_date.max()])
    tpmeans.append(stemp_date.mean())
    tpl1.append(np.linalg.norm(stemp_date.flatten(), 1))
    tpl2.append(np.linalg.norm(stemp_date.flatten(), 2))
    tplinf.append(np.linalg.norm(stemp_date.flatten(), np.inf))

    print(pmeans)
    print(tpmeans)

    for var in vardict.keys():
        var_tiles = np.concatenate((vardict[var][0], vardict[var][1], vardict[var][2], vardict[var][3], vardict[var][4], vardict[var][5]),axis=0).flatten()
        vminmaxes.append([var_tiles.min(),var_tiles.max()])
        vmeans.append(var_tiles.mean())
        vl1.append(np.linalg.norm(var_tiles.flatten(), 1))
        vl2.append(np.linalg.norm(var_tiles.flatten(), 2))
        vlinf.append(np.linalg.norm(var_tiles.flatten(), np.inf))
        # print(var, np.linalg.norm(var_tiles.flatten(), 1))
        print(var, np.mean(np.abs(var_tiles.flatten())))

##################
## Making Plots ##
##################

plotdir = "raw_state"

# Stuff for to make plots more prettier
axwidth=3
axlength=12
fontsize=20
linewidth=6
labelsize=20

plt.rc("text.latex", preamble=r'\boldmath')
plt.rc("text", usetex=True)
# plt.rc("axes", linewidth=axwidth)
# plt.rc(("xtick.major","ytick.major"),size=15)
# plt.rc(("xtick.minor","ytick.minor"),size=8)
# plt.rc(("xtick","ytick"),labelsize='x-large')

# Plot params:
minlon = 1
maxlon = 48
minlat = 1
maxlat = 48

cb_arr = [0.1, 0.05, 0.8, 0.05]
bot_adj = 0.15

plevels0 = 5
plevels1 = 5

tlevels0 = 5
tlevels1 = 5
if args.dir2:
    post = 'diff'
else:
    # plevels0 = [992,994,996,998,1000,1002,1004,1006]
    # plevels1 = [930,940,950,960,970,980,990,1000,1010,1020,1030]

    post = 'range'

tilestrs = [r'$\mathrm{Tile\ 1}$', r'$\mathrm{Tile\ 2}$', r'$\mathrm{Tile\ 3}$', r'$\mathrm{Tile\ 4}$', r'$\mathrm{Tile\ 5}$', r'$\mathrm{Tile\ 6}$']

# Pressure plots
fig1, axs1 = plt.subplots(3,2)
levs = []
for ii in range(3):
    for jj in range(2):
        k = 2*ii+jj
        if k==0:
            cs = axs1[ii,jj].contourf(np.log10(np.abs(ps_plots[k])), plevels0)
            levs = cs.levels
        else:
            cs = axs1[ii,jj].contourf(np.log10(np.abs(ps_plots[k])), levs)
        axs1[ii,jj].annotate(tilestrs[k], (2,2), textcoords='data', size=9)

fig1.subplots_adjust(bottom=bot_adj)
cax = fig1.add_axes(cb_arr)
cbar = fig1.colorbar(cs, cax=cax, orientation='horizontal')

fig1.suptitle(r'$\mathrm{Log_{10}\ Surface\ Pressure}$')
# if args.dir2:
#     fig1.text(0.42,0.91,r'$\mathrm{{log10\ relative\ diff:\ {0}}}$'.format(round(np.log10(np.abs(pmeans[0],2)))))

plt.savefig('{0}/tile_pressure0_{1}.png'.format(plotdir, post))

# Surface Humidity plots
fig2, axs2 = plt.subplots(3,2)
for ii in range(3):
    for jj in range(2):
        k = 2*ii+jj
        if k==0:
            cs = axs2[ii,jj].contourf(np.log10(np.abs(stemp_plots[k])), plevels1)
            levs = cs.levels
        else:
            cs = axs2[ii,jj].contourf(np.log10(np.abs(stemp_plots[k])), levs)
        axs2[ii,jj].annotate(tilestrs[k], (2,2), textcoords='data', size=9)

fig2.subplots_adjust(bottom=bot_adj)
cax = fig2.add_axes(cb_arr)
cbar = fig2.colorbar(cs, cax=cax, orientation='horizontal')

fig2.suptitle(r'$\mathrm{Log_{10}\ Bottom\ Temperature}$')
# if args.dir2:
#     fig2.text(0.42,0.91,r'$\mathrm{{log10\ relative\ diff:\ {0}}}$'.format(round(np.log10(np.abs(tpmeans[0],2)))))

plt.savefig('{0}/tile_bot_temp0_{1}.png'.format(plotdir, post))