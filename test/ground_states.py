import numpy
from beclab import *
from beclab.meters import DensityProfile
from beclab.constants import buildProjectorMask
import itertools
import time


def testGroundState(gpu, comp, grid_type, dim, gs_type, use_cutoff):
	env = envs.cuda() if gpu else envs.cpu()
	try:
		return runTest(env, comp, grid_type, dim, gs_type, use_cutoff)
	finally:
		env.release()

def runTest(env, comp, grid_type, dim, gs_type, use_cutoff):
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
		('3d', 'uniform'): (128, 8, 8),
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

	e_cut = {
		'1d': 8000,
		'3d': 7000
	}[dim]

	# Prepare constants and grid
	constants = Constants(double=env.supportsDouble(),
		e_cut=(e_cut if use_cutoff else None),
		**constants_kwds)
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

	# Create ground state
	t1 = time.time()
	psi = gs.create(target_N)
	t2 = time.time()
	t_gs = t2 - t1

	# check that 2-component stats object works properly
	N_xspace1 = psi.density_meter.getN()
	psi.toMSpace()
	N_mspace = psi.density_meter.getN()
	mode_data = numpy.abs(env.fromDevice(psi.data)) # remember mode data
	psi.toXSpace()

	# population in x-space after double transformation (should not change)
	N_xspace2 = psi.density_meter.getN()

	# calculate energy and chemical potential (per particle)
	E = psi.interaction_meter.getEnergy() / constants.hbar / constants.wz
	mu = psi.interaction_meter.getMu() / constants.hbar / constants.wz
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
	assert mu.shape == (2,)

	if comp == '1comp':
		assert E[1] == 0
	else:
		assert E[1] != 0

	if comp == '1comp':
		assert mu[1] == 0
	else:
		assert mu[1] != 0

	# There should be some difference between analytical mu and numerical one,
	# so it is more of a sanity check
	assert abs(mu.sum() - mu_tf) / mu_tf < 0.35

	# Check that GS is really restricted by mask
	mask = numpy.tile(buildProjectorMask(constants, grid),
		(psi.components, 1) + (1,) * grid.dim)
	masked_mode_data = mode_data * (1.0 - mask)
	assert masked_mode_data.max() < 1e-6 * mode_data.max()

	E = E.sum()
	mu = mu.sum()
	Nx = N_xspace2.sum()
	Nm = N_mspace.sum()

	# Results
	print ("  Modes: {modes_num} out of {total_modes}\n" +
		"  N(x-space) = {Nx:.4f}, N(m-space) = {Nm:.4f}, " +
		"E = {E:.4f} hbar wz, mu = {mu:.4f} hbar wz\n" +
		"  Time spent: {t_gs} s").format(
		Nx=Nx, Nm=Nm, E=E, mu=mu, t_gs=t_gs, modes_num=int(mask.sum()), total_modes=mask.size)

	z = grid.z * 1e6 # cast to micrometers
	profile = prj.getZ(psi) / 1e6 # cast to micrometers^-1

	plots = [
		XYData(str(env) + ", " + grid_type + ", " + gs_type + " |" + str(i + 1) + ">",
			z, profile[i],
			xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0,
			linestyle=('-' if i == 0 else '--'))
		for i in xrange(len(profile))
	]

	return plots[:(1 if comp == '1comp' else 2)]


if __name__ == '__main__':

	prefix = 'ground_states_'
	types = (
		('1d', '3d',),
		(False, True,), # 1- or 2-component
		(False, True,), # cutoff usage
	)
	tests = (
		('uniform', 'harmonic',), # grid type
		('TF', 'split-step', 'rk5',), # ground state type
		(False, True,), # gpu usage
	)

	for dim, two_comp, use_cutoff in itertools.product(*types):

		comp = '2comp' if two_comp else '1comp'
		cutoff = "cutoff" if use_cutoff else "nocutoff"

		print "\n*** {dim} *** ({comp}, {cutoff})\n".format(
			dim=dim, comp=comp, cutoff=cutoff)

		plots_gpu = []
		plots_cpu = []
		for grid_type, gs_type, gpu in itertools.product(*tests):
			if grid_type == 'harmonic' and gs_type == 'split-step':
				continue
			print "* Testing", ", ".join((grid_type, gs_type, ("GPU" if gpu else "CPU")))
			p = testGroundState(gpu, comp, grid_type, dim, gs_type, use_cutoff)
			to_add = plots_gpu if gpu else plots_cpu
			to_add += p

		name = prefix + dim + '_' + comp + '_' + cutoff
		if len(plots_gpu) > 0: XYPlot(plots_gpu).save(name + '_GPU.pdf')
		if len(plots_cpu) > 0: XYPlot(plots_cpu).save(name + '_CPU.pdf')
