import numpy
from beclab import *
from beclab.constants import SOConstants2D
from beclab.constants import _HBAR as hbar
from beclab.ground_state_so import SOGroundStateEvo
import time, copy

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

def getW(psi0, psi1, dV):
	N = ((numpy.abs(psi0) ** 2 + numpy.abs(psi1) ** 2) * dV).sum()
	f0 = psi0 / numpy.sqrt(N)
	f1 = psi1 / numpy.sqrt(N)
	return (((numpy.abs(f0) ** 2 - numpy.abs(f1) ** 2) ** 2 - (2 * (f0 * f1).real) ** 2) * dV).sum()

def buildPTfromP(data0, data1):

	N = (numpy.abs(data0) ** 2).sum() + (numpy.abs(data1) ** 2).sum()

	pt_data0 = spatialInv(data1.conj())
	pt_data1 = spatialInv(data0.conj())

	pt0 = (data0 + pt_data0) / numpy.sqrt(2.0)
	pt1 = (data1 + pt_data1) / numpy.sqrt(2.0)

	new_N = (numpy.abs(pt0) ** 2).sum() + (numpy.abs(pt1) ** 2).sum()
	pt0 *= numpy.sqrt(N / new_N)
	pt1 *= numpy.sqrt(N / new_N)

	return pt0, pt1

def buildPfromPT(data0, data1):

	N = (numpy.abs(data0) ** 2).sum() + (numpy.abs(data1) ** 2).sum()

	p_data0 = spatialInv(data0)
	p_data1 = -spatialInv(data1)

	p0 = (data0 + p_data0) / numpy.sqrt(2.0)
	p1 = (data1 + p_data1) / numpy.sqrt(2.0)

	new_N = (numpy.abs(p0) ** 2).sum() + (numpy.abs(p1) ** 2).sum()
	p0 *= numpy.sqrt(N / new_N)
	p1 *= numpy.sqrt(N / new_N)

	return p0, p1


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
		g11=so_constants.g_intra, g22=so_constants.g_intra, g12=so_constants.g_inter,
		lambda_R=so_constants.lambda_R, m=so_constants.m,
		fx=so_constants.f_perp, fy=so_constants.f_perp,
		e_cut=e_cut)

	box_size = (so_constants.a_perp * area_size, so_constants.a_perp * area_size)
	grid = UniformGrid(env, constants, (grid_size, grid_size), box_size)

	t_scale = 1.0 / (so_constants.f_perp * numpy.pi * 2)
	E_scale = constants.hbar * so_constants.f_perp * numpy.pi * 2
	E_modifier = so_constants.lambda_SO ** 2 / 2.0 * E_scale * so_constants.N
	print "time scale = ", t_scale
	gs = SOGroundStateEvo(env, constants, grid, dt=t_scale * dt_dimensionless, components=2)

	N = so_constants.N
	print "target N = ", N
	t1 = time.time()
	psi = gs.create(N, precision=precision, E_modifier=E_modifier, random_init=random_init)
	print "creation time = ", time.time() - t1

	data = env.fromDevice(psi.data)
	p0, p1 = buildPfromPT(data[0, 0], data[1, 0])
	p_data = numpy.concatenate([p0, p1]).reshape(psi.data.shape)
	psi.data.set(p_data)

	print "second pass"
	t1 = time.time()
	psi = gs.create(N, precision=1e-10, E_modifier=E_modifier, psi=psi)
	print "creation time = ", time.time() - t1

	plot_params = dict(
		xmin=-box_size[1] / 2 / so_constants.a_perp,
		xmax=box_size[1] / 2 / so_constants.a_perp,
		ymin=-box_size[0] / 2 / so_constants.a_perp,
		ymax=box_size[0] / 2 / so_constants.a_perp,
		xname="$x / a_{\\perp}$", yname="$y / a_{\\perp}$", zname="density",
		zmin=0,
		title=(
			"$\\lambda_{{SO}} = {lambda_SO}$, $g_{{\\uparrow\\downarrow}} / g = {g_ratio}$, $g N = {g_strength}$, $k_{{cut}} = {k_cutoff}$\n" +
			"grid: ${gsize}\\times{gsize}$, $dt = {dt} / \\omega_{{\\perp}}$, $precision = {precision}$, " + ("random" if random_init else "uniform") + " initial conditions"
			).format(
				lambda_SO=so_constants.lambda_SO, g_ratio=so_constants.g_ratio, k_cutoff=k_cutoff,
				g_strength=so_constants.g_strength,
				gsize=grid_size, dt=dt_dimensionless, precision=precision,
			)
	)

	def analyzeData(title, suffix, psi):

		print title

		Ns = psi.density_meter.getNTotal()
		print "  final N = ", Ns

		E = psi.interaction_meter.getETotal()
		E_shifted = (E.sum() + E_modifier) / E_scale / so_constants.N
		print "  final energy = ", E_shifted
		data = env.fromDevice(psi.data)
		W = getW(data[0, 0], data[1, 0], grid.dV) * (so_constants.a_perp ** 2)
		n = env.fromDevice(psi.density_meter.getNDensity())
		p, pt = getSymmetries(n)

		n *= so_constants.a_perp ** 2

		results = "$E / N + \\lambda_{{SO}}^2 / 2 = {E:.6f}$, $n_P = {n_P:.4f}$, $n_{{PT}} = {n_PT:.4f}$, $W = {W:.5f}$".format(E=E_shifted, n_P=p, n_PT=pt, W=W)

		params = copy.deepcopy(plot_params)
		params['title'] = title + ": " + plot_params['title'] + "\n" + results
		params['zmax'] = n.max()

		suffix = "{g_ratio}_{g_strength}_{lambda_SO}_k{k_cutoff}_{init}".format(
			g_ratio=so_constants.g_ratio, g_strength=so_constants.g_strength,
			lambda_SO=so_constants.lambda_SO, k_cutoff=k_cutoff,
			init=("random" if random_init else "uniform")) + suffix
		buildPlot(n[0, 0], n[1, 0], 'so_' + suffix + '.pdf', **params)

	analyzeData("Ground state", "", psi)

	data = env.fromDevice(psi.data)

	pt0, pt1 = buildPTfromP(data[0, 0], data[1, 0])
	pt_data = numpy.concatenate([pt0, pt1]).reshape(psi.data.shape)

	p0, p1 = buildPfromPT(data[0, 0], data[1, 0])
	p_data = numpy.concatenate([p0, p1]).reshape(psi.data.shape)

	psi.data.set(pt_data)
	analyzeData("PT-transformed", "_pt", psi)

	psi.data.set(p_data)
	analyzeData("P-transformed", "_p", psi)

	env.release()


if __name__ == '__main__':

#	runTest(15, 1, 1, grid_size=1024, dt_dimensionless=1.0 / 50,
#		area_size=6, random_init=True, k_cutoff=2048)

#	IIB, l=4
#	runTest(1.5, 0.8, 4, grid_size=1024, dt_dimensionless=1.0/400,
#		area_size=4, random_init=True, k_cutoff=512)

#	IIB, l=1
#	runTest(1.5, 35, 1, grid_size=1024, dt_dimensionless=1.0/400,
#		area_size=8, random_init=True, k_cutoff=512)

#	IIA, l=4
#	runTest(0.5, 0.8, 4, grid_size=1024, dt_dimensionless=1.0/400,
#		area_size=4, random_init=True, k_cutoff=512)

#	IIA, l=1
#	runTest(0.5, 35, 1, grid_size=1024, dt_dimensionless=1.0/400,
#		area_size=8, random_init=True, k_cutoff=512)

	runTest(1.5, 60, 1, grid_size=1024, dt_dimensionless=1.0/200, precision=1e-7,
		area_size=10, random_init=True, k_cutoff=512)

#	runTest(1.5, 1.5, 6, grid_size=1024, dt_dimensionless=1.0/800, precision=1e-7,
#		area_size=10, random_init=True, k_cutoff=512)
