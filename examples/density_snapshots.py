"""
Create 2-component ground state and plot various plots of its density
"""

import numpy
from beclab import *
from beclab.meters import ProjectionMeter


N = 20000

env = envs.cuda()
constants = Constants(double=env.supportsDouble())
grid = UniformGrid.forN(env, constants, N, (128, 32, 16))

gs = SplitStepGroundState(env, constants, grid, dt=1e-5)

psi = gs.create((N / 2, N / 2), precision=1e-8)
prj = ProjectionMeter.forPsi(psi)

# Create various density profiles
prefix = "density_snapshots"

# cast grid points to micrometers
x = grid.x * 1e6
y = grid.y * 1e6
z = grid.z * 1e6

# Projection on Z axis
z_proj = prj.getZ(psi) / 1e6 # cast to micrometers^-1

# Projection on XY and YZ planes
xy_proj = prj.getXY(psi) / 1e12 # cast to micrometers^-2
yz_proj = prj.getYZ(psi) / 1e12 # cast to micrometers^-2

# Slice along XY and YZ planes (through the center)
xy_slice = prj.getXYSlice(psi) / 1e18 # cast to micrometers^-3
yz_slice = prj.getYZSlice(psi) / 1e18 # cast to micrometers^-3

env.release()

# Build plots

# Z-projection

XYPlot([
	XYData("|1>", z, z_proj[0],
		xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0),
	XYData("|2>", z, z_proj[1],
		xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0),
]).save(prefix + '_z_proj.pdf')

# Plane projections

HeightmapPlot(HeightmapData(
	"|1>", xy_proj[0],
	xname="X ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-2}$)",
	xmin=x[0], xmax=x[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_xy_proj_1.pdf')

HeightmapPlot(HeightmapData(
	"|2>", xy_proj[1],
	xname="X ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-2}$)",
	xmin=x[0], xmax=x[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_xy_proj_2.pdf')

HeightmapPlot(HeightmapData(
	"|1>", yz_proj[0].transpose(),
	xname="Z ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-2}$)",
	xmin=z[0], xmax=z[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_yz_proj_1.pdf')

HeightmapPlot(HeightmapData(
	"|2>", yz_proj[1].transpose(),
	xname="Z ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-2}$)",
	xmin=z[0], xmax=z[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_yz_proj_2.pdf')

# Slices

HeightmapPlot(HeightmapData(
	"|1>", xy_slice[0],
	xname="X ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-3}$)",
	xmin=x[0], xmax=x[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_xy_slice_1.pdf')

HeightmapPlot(HeightmapData(
	"|2>", xy_slice[1],
	xname="X ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-3}$)",
	xmin=x[0], xmax=x[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_xy_slice_2.pdf')

HeightmapPlot(HeightmapData(
	"|1>", yz_slice[0].transpose(),
	xname="Z ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-3}$)",
	xmin=z[0], xmax=z[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_yz_slice_1.pdf')

HeightmapPlot(HeightmapData(
	"|2>", yz_slice[1].transpose(),
	xname="Z ($\\mu$m)", yname="Y ($\\mu$m)", zname="Density ($\\mu$m$^{-3}$)",
	xmin=z[0], xmax=z[-1], ymin=y[0], ymax=y[-1]
)).save(prefix + '_yz_slice_2.pdf')
