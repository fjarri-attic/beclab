import numpy
from beclab import *
from beclab.constants import SOConstants2D
from beclab.constants import _HBAR as hbar
from beclab.ground_state_so import SOGroundState
from beclab.meters import ParticleStatistics
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

def spatialInv(a):
	return numpy.fliplr(numpy.flipud(a))

def getSymmetries(n):
	na = n[0, 0]
	nb = n[1, 0]
	na_inv = spatialInv(na)
	nb_inv = spatialInv(nb)
	pa = (numpy.abs(na - na_inv)).sum()
	pb = (numpy.abs(nb - nb_inv)).sum()
	pta = (numpy.abs(na - nb_inv)).sum()
	ptb = (numpy.abs(nb - na_inv)).sum()
	Na = na.sum()
	Nb = nb.sum()

	p = (pa / Na + pb / Nb) / 4
	pt = (pta / (Na + Nb) + ptb / (Na + Nb)) / 2
	return p, pt

def buildPlot(na, nb, filename, **plot_params):

	fig = plt.figure(figsize=(8, 5))
	fig.suptitle(plot_params['title'])

	grid = AxesGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.4, label_mode="L", aspect=3)

	params = dict(interpolation='bicubic', origin='lower',
			aspect=1, extent=(plot_params['xmin'], plot_params['xmax'],
				plot_params['ymin'], plot_params['ymax']),
			vmin=plot_params['zmin'], vmax=plot_params['zmax'])
	textparams = dict(horizontalalignment='left', verticalalignment='top',
			bbox=dict(facecolor='white'), fontsize=8)

	im1 = grid[0].imshow(na, **params)
	grid[0].text(0, 1, "spin-up", transform=grid[0].transAxes, **textparams)
	grid[0].set_xlabel(plot_params['xname'])
	grid[0].set_ylabel(plot_params['yname'])
	im2 = grid[1].imshow(nb, **params)
	grid[1].text(0, 1, "spin-down", transform=grid[1].transAxes, **textparams)

	fig.savefig(filename)

def runTest(g_ratio, g_strength, lambda_SO, grid_size=2048, area_size=6,
		precision=1e-7, dt_dimensionless=1.0/50, random_init=True, k_cutoff=1024):

	so_constants = SOConstants2D(g_ratio=g_ratio, g_strength=g_strength, lambda_SO=lambda_SO)
	e_cut = hbar * (k_cutoff / so_constants.a_perp) ** 2 / (2.0 * so_constants.m)

	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(),
		g_intra=so_constants.g_intra, g_inter=so_constants.g_inter,
		lambda_R=so_constants.lambda_R, m=so_constants.m,
		fx=so_constants.f_perp, fy=so_constants.f_perp,
		e_cut=e_cut)

	box_size = (so_constants.a_perp * area_size, so_constants.a_perp * area_size)
	grid = UniformGrid(constants, (grid_size, grid_size), box_size)

	t_scale = 1.0 / (so_constants.f_perp * numpy.pi * 2)
	E_scale = constants.hbar * so_constants.f_perp * numpy.pi * 2
	E_modifier = so_constants.lambda_SO ** 2 / 2 * E_scale
	print "time scale = ", t_scale
	gs = SOGroundState(env, constants, grid, dt=t_scale * dt_dimensionless,
		components=2, precision=precision, E_modifier=E_modifier, random_init=random_init)
	stats = gs._statistics

	N = so_constants.N
	print "target N = ", N
	t1 = time.time()
	psi = gs.create((N / 2, N / 2))
	print "creation time = ", time.time() - t1

	Ns = stats.getN(psi)
	print "final N = ", Ns
	E = stats.getSOEnergy(psi)
	E_shifted = (E.sum() + E_modifier) / E_scale
	print "final energy = ", E_shifted
	n = env.fromDevice(stats.getDensity(psi))
	p, pt = getSymmetries(n)
	env.release()

	n *= so_constants.a_perp ** 2
	plot_params = dict(
		xmin=-box_size[1] / 2 / so_constants.a_perp,
		xmax=box_size[1] / 2 / so_constants.a_perp,
		ymin=-box_size[0] / 2 / so_constants.a_perp,
		ymax=box_size[0] / 2 / so_constants.a_perp,
		xname="$x / a_{\\perp}$", yname="$y / a_{\\perp}$", zname="density",
		zmin=0, zmax=n.max(),
		title=(
			"$\\lambda_{{SO}} = {lambda_SO}$, $g_{{\\uparrow\\downarrow}} / g = {g_ratio}$, $g N = {g_strength}$, $k_{cut} = {k_cutoff}$\n" +
			"grid: ${gsize}\\times{gsize}$, $dt = {dt} / \\omega_{{\\perp}}$, $precision = {precision}$, " + ("random" if random_init else "uniform") + " initial conditions\n" +
			"$E / N + \\lambda_{{SO}}^2 / 2 = {E:.6f}$, $n_P = {n_P:.4f}$, $n_{{PT}} = {n_PT:.4f}$"
			).format(
				lambda_SO=so_constants.lambda_SO, g_ratio=so_constants.g_ratio, k_cutoff=k_cutoff,
				g_strength=so_constants.g_strength,
				gsize=grid_size, dt=dt_dimensionless, precision=precision,
				E=E_shifted, n_P=p, n_PT=pt
			)
	)

	suffix = "{g_ratio}_{g_strength}_{lambda_SO}_k{k_cutoff}_{init}".format(
		g_ratio=so_constants.g_ratio, g_strength=so_constants.g_strength,
		lambda_SO=so_constants.lambda_SO, k_cutoff=k_cutoff,
		init=("random" if random_init else "uniform"))
	buildPlot(n[0, 0], n[1, 0], 'so_' + suffix + '.pdf', **plot_params)

if __name__ == '__main__':
	for k_cutoff in (256, 512):
		runTest(1.5, 0.2, 20, grid_size=1024, dt_dimensionless=1.0/50,
			area_size=4, random_init=False, k_cutoff=k_cutoff)
