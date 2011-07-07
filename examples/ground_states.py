import numpy
from beclab import *
from beclab.meters import DensityProfile, ParticleStatistics
import itertools
import time


def testThomasFermi(gpu, grid_type, dim, gs_type):
	"""
	Creates Thomas-Fermi ground state using different types of representations
	"""

	# Prepare constants and environment

	if dim != '3d':
		parameters = dict(use_effective_area=True, fx=42e3, fy=42e3, fz=90)
	else:
		parameters = {}

	env = envs.cuda() if gpu else envs.cpu()
	constants = Constants(double=False if gpu else True, **parameters)
	N = 50000 if dim == '3d' else 60 # number of atoms

	if grid_type == 'uniform':
		shape = (128, 16, 16) if dim == '3d' else (64,)
		grid = UniformGrid.forN(env, constants, N, shape)
	elif grid_type == 'harmonic':
		shape = (50, 10, 10) if dim == '3d' else (50,)
		grid = HarmonicGrid(env, constants, shape)

	# Prepare 'apparatus'

	if gs_type == "TF":
		gs = TFGroundState(env, constants, grid)
	elif gs_type == "split-step":
		gs = SplitStepGroundState(env, constants, grid)

	prj = DensityProfile(env, constants, grid)
	stats = ParticleStatistics(env, constants, grid)

	# Create ground state

	t1 = time.time()
	cloud = gs.createCloud(N)
	t2 = time.time()
	t_gs = t2 - t1

	psi = cloud.psi0

	# population in x-space
	N_xspace1 = stats.getN(psi)

	# population in mode space
	psi.toMSpace()
	N_mspace = stats.getN(psi)
	psi.toXSpace()

	# population in x-space after double transformation (should not change)
	N_xspace2 = stats.getN(psi)

	E = stats.getEnergy(psi) / constants.hbar / constants.wz
	mu = stats.getMu(psi) / constants.hbar / constants.wz
	mu_tf = constants.muTF(N, dim=grid.dim) / constants.hbar / constants.wz

	print ("N(x-space) = {Nx}, N(m-space) = {Nm},\n" +
		"E = {E} hbar w_z, mu = {mu} hbar w_z (mu_analytical = {mu_tf})\n" +
		"Time spent: {t_gs} s").format(
		Nx=N_xspace2, Nm=N_mspace, E=E, mu=mu, mu_tf=mu_tf, t_gs=t_gs)

	z = grid.z * 1e6 # cast to micrometers
	profile = prj.getZ(psi) / 1e6 # cast to micrometers^-1
	plot = XYData(str(env) + ", " + grid_type + ", " + gs_type,
		z, profile,
		xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0)

	env.release()

	# Checks

	# check that double transform did not change N
	assert abs(N_xspace1 - N_xspace2) / N_xspace2 < 1e-6
	# check that we actually got N we asked for
	assert abs(N_xspace2 - N) / N < 1e-6 if constants.double else 1e-4
	# There is certain difference for harmonic grid,
	# because FHT does not conserve population (TODO: prove it mathematically)
	assert abs(N_mspace - N_xspace2) / N_xspace2 < 2e-2 if grid_type == 'harmonic' else 1e-6

	# There should be some difference between analytical mu and numerical one,
	# but it shouldn't be very big
	assert abs(mu - mu_tf) / mu_tf < 2e-2

	return plot

if __name__ == '__main__':

	prefix = 'ground_states_'
	tests = (
		(False, True), # gpu usage
		('uniform', 'harmonic'), # grid type
		('TF',) # ground state type
	)

	# Thomas-Fermi ground states
	for dim in ('1d', '3d'):
		print
		print "***", dim, "***"
		print
		plots = []
		for gpu, grid_type, gs_type in itertools.product(*tests):
			if grid_type == 'harmonic' and gs_type == 'split-step':
				continue
			print "* Testing", grid_type, "grid and", gs_type, "on", ("GPU" if gpu else "CPU")
			plots.append(testThomasFermi(gpu, grid_type, dim, gs_type))
		XYPlot(plots).save(prefix + dim + '.pdf')
