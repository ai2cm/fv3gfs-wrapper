import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from argparse import ArgumentParser

usage = "usage: python %(prog)s <output directory> [optional 2nd output directory] [other options]"
parser = ArgumentParser(usage=usage)

parser.add_argument(
    "dir1", type=str, action="store", help="directory containing outputs to plot"
)
parser.add_argument(
    "dir2",
    type=str,
    action="store",
    help="directory containing outputs to compare to",
    nargs="?",
)
args = parser.parse_args()

jw_colors = [
    "#ec1b8c",
    "#a6228e",
    "#20419a",
    "#0085cc",
    "#03aeef",
    "#03aa4f",
    "#c8da2c",
    "#fff200",
    "#f99e1c",
    "#ed1c24",
]

##################
# Data Wrangling #
##################

files = [
    r"atmos_4xdaily_fine_inst.tile1.nc",
    r"atmos_4xdaily_fine_inst.tile2.nc",
    r"atmos_4xdaily_fine_inst.tile3.nc",
    r"atmos_4xdaily_fine_inst.tile4.nc",
    r"atmos_4xdaily_fine_inst.tile5.nc",
    r"atmos_4xdaily_fine_inst.tile6.nc",
]

tplot = [4, 6, 8, 10]

lats = []
lons = []
ps_plots = []
t850_plots = []

for f in files:
    fname = args.dir1 + f
    ncfile = Dataset(fname, "r")
    nc_attrs = ncfile.ncattrs()
    nc_dims = [dim for dim in ncfile.dimensions]  # list of nc dimensions
    nc_vars = [var for var in ncfile.variables]  # list of nc variables

    lat = ncfile.variables["grid_yt"][:]
    lon = ncfile.variables["grid_xt"][:]
    time = ncfile.variables["time"][:]
    pl = ncfile.variables["plev"][:]
    ps = ncfile.variables["ps"][:] / 100.0  # convert to hPa
    temp = ncfile.variables["t_plev"][:]

    temp_850 = temp[:, pl == 850, :, :][:, 0, :, :]  # temperature at 850 hPa

    ps_plot = np.array([ps[time == t] for t in tplot])
    t850_plot = np.array([temp_850[time == t] for t in tplot])

    if args.dir2:
        fname2 = args.dir2 + f
        ncf2 = Dataset(fname2, "r")
        lat2 = ncfile.variables["grid_yt"][:]
        lon2 = ncfile.variables["grid_xt"][:]
        time2 = ncfile.variables["time"][:]
        pl2 = ncfile.variables["plev"][:]
        ps2 = ncfile.variables["ps"][:] / 100.0  # convert to hPa
        temp2 = ncfile.variables["t_plev"][:]
        temp2_850 = temp[:, pl == 850, :, :][:, 0, :, :]  # temperature at 850 hPa
        ps_plot2 = np.array([ps[time == t] for t in tplot])
        t850_plot2 = np.array([temp_850[time == t] for t in tplot])

        t850_diff = t850_plot - t850_plot2
        ps_diff = ps_plot - ps_plot2

        ps_plots.append(ps_diff)
        t850_plots.append(t850_diff)
    else:
        ps_plots.append(ps_plot)
        t850_plots.append(t850_plot)

    lats.append(lat)
    lons.append(lon)


####################
# Doin' some stats #
####################
if args.dir2:
    pminmaxes = []
    pmeans = []
    pl1 = []
    pl2 = []
    plinf = []

    tminmaxes = []
    tmeans = []
    tl1 = []
    tl2 = []
    tlinf = []

    p = np.array(ps_plots)
    tarr = np.array(t850_plots)
    for tt in range(len(tplot)):
        pres_date = np.concatenate(
            (
                p[0, tt, :, :][0],
                p[1, tt, :, :][0],
                p[2, tt, :, :][0],
                p[3, tt, :, :][0],
                p[4, tt, :, :][0],
                p[5, tt, :, :][0],
            ),
            axis=0,
        )
        temp_date = np.concatenate(
            (
                tarr[0, tt, :, :][0],
                tarr[1, tt, :, :][0],
                tarr[2, tt, :, :][0],
                tarr[3, tt, :, :][0],
                tarr[4, tt, :, :][0],
                tarr[5, tt, :, :][0],
            ),
            axis=0,
        )

        pminmaxes.append([pres_date.min(), pres_date.max()])
        pmeans.append(pres_date.mean())
        pl1.append(np.linalg.norm(pres_date, 1))
        pl2.append(np.linalg.norm(pres_date, 2))
        plinf.append(np.linalg.norm(pres_date, np.inf))

        tminmaxes.append([temp_date.min(), temp_date.max()])
        tmeans.append(temp_date.mean())
        tl1.append(np.linalg.norm(temp_date, 1))
        tl2.append(np.linalg.norm(temp_date, 2))
        tlinf.append(np.linalg.norm(temp_date, np.inf))

    print(pmeans)
    print(tmeans)

################
# Making Plots #
################

# Stuff for to make plots more prettier
axwidth = 3
axlength = 12
fontsize = 20
linewidth = 6
labelsize = 20

plt.rc("text.latex", preamble=r"\boldmath")
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

if args.dir2:
    plevels0 = 5
    plevels1 = 5

    tlevels0 = 5
    tlevels1 = 5
    post = "diff"
else:
    plevels0 = [992, 994, 996, 998, 1000, 1002, 1004, 1006]
    plevels1 = [930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030]

    tlevels0 = [220, 230, 240, 250, 260, 270, 280, 290, 300, 310]
    tlevels1 = [220, 230, 240, 250, 260, 270, 280, 290, 300, 310]
    post = "range"

tilestrs = [
    r"$\mathrm{Tile\ 1}$",
    r"$\mathrm{Tile\ 2}$",
    r"$\mathrm{Tile\ 3}$",
    r"$\mathrm{Tile\ 4}$",
    r"$\mathrm{Tile\ 5}$",
    r"$\mathrm{Tile\ 6}$",
]

# Pressure plots
# day4
fig1, axs1 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs1[ii, jj].contourf(lons[k], lats[k], ps_plots[k][0][0], plevels0)
        axs1[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig1.subplots_adjust(bottom=bot_adj)
cax = fig1.add_axes(cb_arr)
cbar = fig1.colorbar(cs, cax=cax, orientation="horizontal")

fig1.suptitle(r"$\mathrm{Pressure\ day\ 4}$")
if args.dir2:
    fig1.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(pmeans[0], 2)))

plt.savefig("tile_pressure0_{0}.png".format(post))


# day6
fig2, axs2 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs2[ii, jj].contourf(lons[k], lats[k], ps_plots[k][1][0], plevels0)
        axs2[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig2.subplots_adjust(bottom=bot_adj)
cax = fig2.add_axes(cb_arr)
cbar = fig2.colorbar(cs, cax=cax, orientation="horizontal")

fig2.suptitle(r"$\mathrm{Pressure\ day\ 6}$")
if args.dir2:
    fig2.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(pmeans[1], 2)))

plt.savefig("tile_pressure1_{0}.png".format(post))


# day8
fig3, axs3 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs3[ii, jj].contourf(lons[k], lats[k], ps_plots[k][2][0], plevels1)
        axs3[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig3.subplots_adjust(bottom=bot_adj)
cax = fig3.add_axes(cb_arr)
cbar = fig3.colorbar(cs, cax=cax, orientation="horizontal")

fig3.suptitle(r"$\mathrm{Pressure\ day\ 8}$")
if args.dir2:
    fig3.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(pmeans[2], 2)))

plt.savefig("tile_pressure2_{0}.png".format(post))


# day10
fig4, axs4 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs4[ii, jj].contourf(lons[k], lats[k], ps_plots[k][3][0], plevels1)
        axs4[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig4.subplots_adjust(bottom=bot_adj)
cax = fig4.add_axes(cb_arr)
cbar = fig4.colorbar(cs, cax=cax, orientation="horizontal")

fig4.suptitle(r"$\mathrm{Pressure\ day\ 10}$")
if args.dir2:
    fig4.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(pmeans[3], 2)))

plt.savefig("tile_pressure3_{0}.png".format(post))


# Temperature plots
# day4
fig5, axs5 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs5[ii, jj].contourf(lons[k], lats[k], t850_plots[k][0][0], tlevels0)
        axs5[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig5.subplots_adjust(bottom=bot_adj)
cax = fig5.add_axes(cb_arr)
cbar = fig5.colorbar(cs, cax=cax, orientation="horizontal")

fig5.suptitle(r"$\mathrm{Temperature\ day\ 4}$")
if args.dir2:
    fig5.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(tmeans[0], 2)))

plt.savefig("tile_temp0_{0}.png".format(post))


# day6
fig6, axs6 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs6[ii, jj].contourf(lons[k], lats[k], t850_plots[k][1][0], tlevels0)
        axs6[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig6.subplots_adjust(bottom=bot_adj)
cax = fig6.add_axes(cb_arr)
cbar = fig6.colorbar(cs, cax=cax, orientation="horizontal")

fig6.suptitle(r"$\mathrm{Temperature\ day\ 6}$")
if args.dir2:
    fig6.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(tmeans[1], 2)))

plt.savefig("tile_temp1_{0}.png".format(post))


# day8
fig7, axs7 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs7[ii, jj].contourf(lons[k], lats[k], t850_plots[k][2][0], tlevels1)
        axs7[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig7.subplots_adjust(bottom=bot_adj)
cax = fig7.add_axes(cb_arr)
cbar = fig7.colorbar(cs, cax=cax, orientation="horizontal")

fig7.suptitle(r"$\mathrm{Temperature\ day\ 8}$")
if args.dir2:
    fig7.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(tmeans[2], 2)))

plt.savefig("tile_temp2_{0}.png".format(post))


# day10
fig8, axs8 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        cs = axs8[ii, jj].contourf(lons[k], lats[k], t850_plots[k][3][0], tlevels1)
        axs8[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig8.subplots_adjust(bottom=bot_adj)
cax = fig8.add_axes(cb_arr)
cbar = fig8.colorbar(cs, cax=cax, orientation="horizontal")

fig8.suptitle(r"$\mathrm{Temperature\ day\ 10}$")
if args.dir2:
    fig8.text(0.42, 0.91, r"$\mathrm{{Mean\ diff:\ {0}}}$".format(round(tmeans[3], 2)))

plt.savefig("tile_temp3_{0}.png".format(post))
