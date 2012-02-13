"""
This example demonstrates usage of static per-component potentials
"""

import numpy
from beclab import *

# parameters from Riedel at al. (2010)
N = 1250
f_rad = 109
f_ax = 500
f_rabi = 2100
f_detuning = -40
potentials_separation = 0.52e-6
splitting_time = 12.7e-3

def split_potentials(constants, grid):

	x = grid.x_full
	y = grid.y_full
	z = grid.z_full

	potentials = lambda dz: constants.m * (
		(constants.wx * x) ** 2 +
		(constants.wy * y) ** 2 +
		(constants.wz * (z + dz)) ** 2) / (2.0 * constants.hbar)

	return numpy.concatenate([
		potentials(potentials_separation / 2),
		potentials(-potentials_separation / 2)
	]).reshape((2,) + x.shape).astype(constants.scalar.dtype)


env = envs.cuda()
constants = Constants(double=env.supportsDouble(),
	a11=100.4, a12=97.7, a22=95.0, fx=f_ax, fy=f_ax, fz=f_rad)

# axial size of the N / 2 cloud ~ 8e-6 >> potentials_separation
# therefore we can safely use normal grid, provided that it has big enough border
grid = UniformGrid.forN(env, constants, N, (128, 8, 8), border=2)

gs = SplitStepGroundState(env, constants, grid, dt=1e-6)
pulse = Pulse(env, constants, grid, f_rabi=f_rabi, f_detuning=f_detuning)
evolution = SplitStepEvolution(env, constants, grid,
	potentials=split_potentials(constants, grid),
	dt=1e-6)

v = VisibilityCollector(env, constants, grid)
#a = AxialProjectionCollector(env, constants, grid)
u = UncertaintyCollector(env, constants, grid)

psi = gs.create((N, 0))
psi.toWigner(256)
pulse.apply(psi, numpy.pi / 2)
evolution.run(psi, splitting_time, callbacks=[v, u], callback_dt=splitting_time / 100)
env.synchronize()
env.release()

#times, heightmap = a.getData()
#HeightmapPlot(
#	HeightmapData("test", heightmap,
#		xmin=0, xmax=splitting_time * 1e3,
#		ymin=grid.z[0] * 1e6,
#		ymax=grid.z[-1] * 1e6,
#		zmin=-1, zmax=1,
#		xname="T (ms)", yname="z ($\\mu$m)", zname="Spin projection")
#).save('split_potentials_axial.pdf')

times, n_stddev, xi_squared = u.getData()
XYPlot([
	XYData("Squeezing", times * 1000, numpy.log10(xi_squared),
		xname="T (ms)", yname="log$_{10}$($\\xi^2$)")
]).save('split_potentials_xi_squared.pdf')

times, vis = v.getData()
XYPlot([
	XYData('test', times * 1e3, vis,
		xname="T (ms)", yname="$\\mathcal{V}$",
		ymin=0, ymax=1)
]).save('split_potentials_vis.pdf')