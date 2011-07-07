import numpy
from beclab import *
from beclab.meters import DensityProfile, ParticleStatistics

eps = 1e-6

def testThomasFermi(gpu, grid_type, dim):
	"""
	Creates Thomas-Fermi ground state using different types of representations
	"""
	if dim != '3d':
		parameters = dict(use_effective_area=True, fx=42e3, fy=42e3, fz=90)
	else:
		parameters = {}

	env = envs.cuda() if gpu else envs.cpu()
	constants = Constants(double=False if gpu else True, **parameters)
	N = 50000 if dim == '3d' else 60 # number of atoms

	if grid_type == 'uniform':
		shape = (64, 8, 8) if dim == '3d' else (64,)
		grid = UniformGrid.forN(env, constants, N, shape)
	elif grid_type == 'harmonic':
		shape = (50, 10, 10) if dim == '3d' else (50,)
		grid = HarmonicGrid(env, constants, shape)

	tf = TFGroundState(env, constants, grid)
	prj = DensityProfile(env, constants, grid)
	stats = ParticleStatistics(env, constants, grid)

	psi = tf.create(N)

	# population in x-space
	N_xspace1 = stats.getN(psi)

	# population in mode space
	psi.toMSpace()
	N_mspace = stats.getN(psi)
	psi.toXSpace()

	# population in x-space after double transformation (should not change)
	N_xspace2 = stats.getN(psi)

	print "N(x-space) = {Nx}, N(m-space) = {Nm}".format(Nx=N_xspace2, Nm=N_mspace)

	z = grid.z
	profile = prj.getZ(psi) / 1e6
	plot = XYData(str(env) + ", " + grid_type, z * 1e6, profile,
		xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0)

	env.release()

	# Checks

	# check that double transform did not change N
	assert abs(N_xspace1 - N_xspace2) / N_xspace2 < 1e-6
	# check that we actually got N we asked for
	assert abs(N_xspace2 - N) / N < 1e-6
	# There is certain difference for harmonic grid,
	# because FHT does not conserve population (TODO: prove it mathematically)
	assert abs(N_mspace - N_xspace2) / N_xspace2 < 2e-2 if grid_type == 'harmonic' else 1e-6

	return plot

if __name__ == '__main__':

	prefix = 'ground_states_'

	# Thomas-Fermi ground states
	for dim in ('1d', '3d'):
		plots = []
		for gpu in (False, True):
			for grid_type in ('uniform', 'harmonic'):
				print "* Testing", grid_type, "on", ("GPU" if gpu else "CPU")
				plots.append(testThomasFermi(gpu, grid_type, dim))
		XYPlot(plots).save(prefix + dim + '_TF.pdf')
