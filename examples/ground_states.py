import numpy
from beclab import *
from beclab.meters import DensityProfile, ParticleStatistics

eps = 1e-6

def testThomasFermi(gpu, grid_type):
	"""
	Creates Thomas-Fermi ground state using different types of representations
	"""

	env = envs.cuda() if gpu else envs.cpu()
	constants = Constants(double=False if gpu else True)
	N = 50000 # number of atoms

	if grid_type == 'uniform':
		grid = UniformGrid.forN(env, constants, N, (64, 8, 8))
	elif grid_type == 'harmonic':
		grid = HarmonicGrid(env, constants, (50, 10, 10))

	tf = TFGroundState(env, constants, grid)
	prj = DensityProfile(env, constants, grid)
	stats = ParticleStatistics(env, constants, grid)

	# check that total population is correct
	psi = tf.create(N)
	N_xspace = stats.getN(psi)
	print "Total population:", N_xspace

	# test normalization in mode space
	psi.toMSpace()
	N_mspace = stats.getN(psi)
	print "Total population in mode space:", N_mspace
	psi.toXSpace()

	z = grid.z
	profile = prj.getZ(psi) / 1e6
	plot = XYData(str(env) + ", " + grid_type, z * 1e6, profile,
		xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0)

	env.release()
	return plot

if __name__ == '__main__':

	prefix = 'ground_states_'

	# Thomas-Fermi ground states
	plots = []
	for gpu in (False, True):
		for grid_type in ('uniform', 'harmonic'):
			plots.append(testThomasFermi(gpu, grid_type))
	XYPlot(plots).save(prefix + 'TF.pdf')
