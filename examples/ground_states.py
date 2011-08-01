import numpy
from beclab import *
from beclab.meters import DensityProfile, ParticleStatistics
import itertools
import time


def testGroundState(gpu, comp, grid_type, dim, gs_type):
	env = envs.cuda() if gpu else envs.cpu()
	try:
		return runTest(env, comp, grid_type, dim, gs_type)
	finally:
		env.release()

def runTest(env, comp, grid_type, dim, gs_type):
	"""
	Creates Thomas-Fermi ground state using different types of representations
	"""

	# additional parameters
	constants_kwds = {
		'1d': dict(use_effective_area=True, fx=42e3, fy=42e3, fz=90),
		'3d': {}
	}[dim]

	# total number of atoms in ground state
	total_N = {
		'1d': 60,
		'3d': 50000
	}[dim]

	# number of lattice points
	shape = {
		('1d', 'uniform'): (64,),
		('3d', 'uniform'): (64, 8, 8),
		('1d', 'harmonic'): (50,),
		('3d', 'harmonic'): (50, 10, 10)
	}[(dim, grid_type)]

	# time step for split-step propagation
	ss_dt = {
		'1d': 1e-6,
		'3d': 1e-5
	}[dim]

	# absolute precision for split-step algorithm
	ss_precision = {
		'1comp': 1e-6,
		'2comp': 1e-8
	}[comp]

	# precision divided by time step for RK5 propagation
	rk5_rprecision = {
		('1d', '1comp'): 1,
		('3d', '1comp'): 1,
		('1d', '2comp'): 1e-2,
		('3d', '2comp'): 1e-3
	}[(dim, comp)]

	# relative error tolerance for RK5 propagation
	rk5_rtol = {
		('1d', 'uniform'): 1e-9,
		('1d', 'harmonic'): 1e-7,
		('3d', 'uniform'): 1e-6,
		('3d', 'harmonic'): 1e-6
	}[(dim, grid_type)]

	# absolute error tolerance divided by atom number for RK5 propagation
	rk5_atol_coeff = 1e-3

	target_N = {
		'1comp': (total_N, 0),
		'2comp': (int(total_N / 2 + 0.05 * total_N), int(total_N / 2 - 0.05 * total_N))
	}[comp]

	# Prepare constants and grid
	constants = Constants(double=env.supportsDouble(), **constants_kwds)
	if grid_type == 'uniform':
		grid = UniformGrid.forN(env, constants, total_N, shape)
	elif grid_type == 'harmonic':
		grid = HarmonicGrid(env, constants, shape)

	# Prepare 'apparatus'

	args = (env, constants, grid)

	if gs_type == "TF":
		gs = TFGroundState(*args)
	elif gs_type == "split-step":
		gs = SplitStepGroundState(*args, precision=ss_precision, dt=ss_dt)
	elif gs_type == "rk5":
		params = dict(eps=rk5_rtol, Nscale=total_N, atol_coeff=rk5_atol_coeff,
			relative_precision=rk5_rprecision)
		if grid_type == 'uniform':
			gs = RK5IPGroundState(*args, **params)
		elif grid_type == 'harmonic':
			gs = RK5HarmonicGroundState(*args, **params)

	prj = DensityProfile(*args)
	stats = ParticleStatistics(*args, components=2)

	# Create ground state
	t1 = time.time()
	psi = gs.create(target_N)
	t2 = time.time()
	t_gs = t2 - t1

	# check that 2-component stats object works properly
	N_xspace1 = stats.getN(psi)
	psi.toMSpace()
	N_mspace = stats.getN(psi)
	psi.toXSpace()

	# population in x-space after double transformation (should not change)
	N_xspace2 = stats.getN(psi)

	# calculate energy and chemical potential (per particle)
	E = stats.getEnergy(psi) / constants.hbar / constants.wz
	mu = stats.getMu(psi) / constants.hbar / constants.wz
	mu_tf = numpy.array(
		[constants.muTF(N, dim=grid.dim) for N in target_N]
	).sum() / constants.hbar / constants.wz

	# Checks
	norm = numpy.linalg.norm
	target_norm = norm(numpy.array(target_N))

	# check that number of particles right after GS creation is correct
	assert N_xspace1.shape == (2,)
	assert norm(N_xspace1 - numpy.array(target_N)) / target_norm < 1e-6

	# check that double transform did not change N
	assert norm(N_xspace1 - N_xspace2) / target_norm < 1e-6

	# TODO: find out what causes difference even for uniform grid
	assert norm(N_mspace - N_xspace2) / target_norm < 0.02

	assert E.shape == (2,)
	if comp == '1comp': assert E[1] == 0
	assert mu.shape == (2,)
	if comp == '1comp': assert mu[1] == 0

	# There should be some difference between analytical mu and numerical one,
	# so it is more of a sanity check
	assert abs(mu.sum() - mu_tf) / mu_tf < 0.35

	E = E.sum()
	mu = mu.sum()
	Nx = N_xspace2.sum()
	Nm = N_mspace.sum()

	# Results

	print ("  N(x-space) = {Nx:.4f}, N(m-space) = {Nm:.4f}, " +
		"E = {E:.4f} hbar wz, mu = {mu:.4f} hbar wz\n" +
		"  Time spent: {t_gs} s").format(
		Nx=Nx, Nm=Nm, E=E, mu=mu, t_gs=t_gs)

	z = grid.z * 1e6 # cast to micrometers
	profile = prj.getZ(psi) / 1e6 # cast to micrometers^-1

	plots = [
		XYData(str(env) + ", " + grid_type + ", " + gs_type + " |" + str(i + 1) + ">",
			z, profile[i],
			xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0,
			linestyle=('-' if i == 0 else '--'))
		for i in xrange(len(profile))
	]

	return plots


if __name__ == '__main__':

	prefix = 'ground_states_'
	types = (
		('1d', '3d'),
		(False, True) # 1- or 2-component
	)
	tests = (
		('uniform', 'harmonic',), # grid type
		('TF', 'split-step', 'rk5',), # ground state type
		(False, True), # gpu usage
	)

	for dim, two_comp in itertools.product(*types):

		print "\n*** {dim} *** ({comp})\n".format(
			dim=dim, comp=('2 components' if two_comp else '1 component'))

		comp = '2comp' if two_comp else '1comp'

		plots_gpu = []
		plots_cpu = []
		for grid_type, gs_type, gpu in itertools.product(*tests):
			if grid_type == 'harmonic' and gs_type == 'split-step':
				continue
			print "* Testing", grid_type, "grid and", gs_type, "on", ("GPU" if gpu else "CPU")
			p = testGroundState(gpu, comp, grid_type, dim, gs_type)
			to_add = plots_gpu if gpu else plots_cpu
			to_add += p

		XYPlot(plots_gpu).save(prefix + dim + '_' + comp + '_GPU.pdf')
		XYPlot(plots_cpu).save(prefix + dim + '_' + comp + '_CPU.pdf')
