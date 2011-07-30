import numpy
import time
import itertools

from beclab import *
from beclab.meters import ParticleStatistics


def testAxialView(gpu, grid_type, dim, prop_type):
	env = envs.cuda() if gpu else envs.cpu()
	try:
		return runTest(env, grid_type, dim, prop_type)
	finally:
		env.release()

def runTest(env, grid_type, dim, prop_type):

	# additional parameters
	constants_kwds = {
		'1d': dict(use_effective_area=True, fx=42e3, fy=42e3, fz=90),
		'3d': {}
	}[dim]

	# total number of atoms in ground state
	total_N = {
		'1d': 60,
		'3d': 150000
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

	constants = Constants(double=env.supportsDouble(), **constants_kwds)
	grid = UniformGrid.forN(env, constants, total_N, shape)

	gs = SplitStepGroundState(env, constants, grid, dt=ss_dt)
	evolution = SplitStepEvolution(env, constants, grid, dt=4e-6)
	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350)
	a = AxialProjectionCollector(env, constants, grid, matrix_pulse=True, pulse=pulse)
	p = ParticleNumberCollector(env, constants, grid, matrix_pulse=True, pulse=pulse)
	v = VisibilityCollector(env, constants, grid)

	# experiment
	psi = gs.create((total_N, 0))

	pulse.apply(psi, theta=0.5 * numpy.pi, matrix=True)

	t1 = time.time()
	evolution.run(psi, time=0.1, callbacks=[a, p, v], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, heightmap = a.getData()

	res = HeightmapPlot(
		HeightmapData("test", heightmap,
			xmin=0, xmax=100,
			ymin=grid.z[0] * 1e6,
			ymax=grid.z[-1] * 1e6,
			zmin=-1, zmax=1,
			xname="T (ms)", yname="z ($\\mu$m)", zname="Spin projection")
	)

	times, Ns, Ntotals = p.getData()
	times, vis = v.getData()

	print "  Final N: ", Ns[:,-1], "(", Ntotals[-1], ")", ", V: ", vis[-1]

	return res


if __name__ == '__main__':

	prefix = 'axial_view_'

	tests = (
		('uniform',), # grid type
		('split-step',), # propagation type
		(False, True,), # gpu usage
	)

	for dim in ('1d', '3d',):
		print "\n*** {dim} ***\n".format(dim=dim)

		for grid_type, prop_type, gpu in itertools.product(*tests):

			if grid_type == 'harmonic' and prop_type == 'split-step':
				continue

			print "* Testing", grid_type, "grid and", prop_type, "on", ("GPU" if gpu else "CPU")
			testAxialView(gpu, grid_type, dim, prop_type).save(
				prefix + dim + '_' + grid_type + '_' + prop_type +
				('_GPU' if gpu else '_CPU') + '.pdf')
