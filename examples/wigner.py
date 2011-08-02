import numpy
import time
import itertools

from beclab import *
from beclab.meters import ParticleStatistics
from beclab.wavefunction import WavefunctionSet

numpy.random.seed(123)

def testWigner(gpu, grid_type, dim, prop_type, repr_type):
	env = envs.cuda() if gpu else envs.cpu()
	try:
		return runTest(env, grid_type, dim, prop_type, repr_type)
	finally:
		env.release()

def runTest(env, grid_type, dim, prop_type, repr_type):

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

	# time steps for split-step GS
	ss_gs_dt = {
		'1d': 1e-6,
		'3d': 1e-5
	}[dim]

	# time steps for split-step propagation
	ss_dt = {
		'1d': 5e-6,
		'3d': 1e-5,
	}[dim]

	ensembles = 16

	wigner = (repr_type == 'wigner')

	constants = Constants(double=env.supportsDouble(), **constants_kwds)
	grid = UniformGrid.forN(env, constants, total_N, shape)

	gs = SplitStepGroundState(env, constants, grid, dt=ss_gs_dt)
	evolution = SplitStepEvolution(env, constants, grid, dt=ss_dt)
	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350)

	# experiment
	psi = gs.create((total_N, 0))
	#psi = WavefunctionSet(env, constants, grid, components=2, ensembles=1)
	if wigner:
		psi.toWigner(ensembles)

	p = ParticleNumberCollector(env, constants, grid, pulse=pulse, verbose=True)
	v = VisibilityCollector(env, constants, grid, verbose=True)

	pulse.apply(psi, theta=0.5 * numpy.pi)

	t1 = time.time()
	evolution.run(psi, time=0.2, callbacks=[p, v], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()

	times, vis = v.getData()
	times, Ns, Ntotals = p.getData()

	print "  N = {N:.4f}, V = {V:.4f}\n  Time spent: {t}".format(
		ss_dt=ss_dt, N=Ntotals[-1], V=vis[-1], t=t2-t1)

	name = ", ".join((repr_type, grid_type, prop_type))
	vis_plot = XYData(
		name, times * 1e3, vis,
		xname="T (ms)", yname="$\\mathcal{V}$", ymin=0, ymax=1)

	return vis_plot


if __name__ == '__main__':

	prefix = 'wigner_'

	tests = (
		('uniform',), # grid type
		('split-step',), # propagation type
		(
		'classical',
		'wigner',
		), # representation type
		(
		True,
		#False,
		), # GPU usage
	)

	for dim in ('3d',):
		print "\n*** {dim} ***\n".format(dim=dim)

		plots_cpu = []
		plots_gpu = []
		for grid_type, prop_type, repr_type, gpu in itertools.product(*tests):

			if grid_type == 'harmonic' and prop_type == 'split-step':
				continue

			print "* Testing " + ", ".join((repr_type, grid_type, prop_type)) + \
				" on " + ("GPU" if gpu else "CPU")

			plot = testWigner(gpu, grid_type, dim, prop_type, repr_type)
			to_add = plots_gpu if gpu else plots_cpu
			to_add.append(plot)

		if len(plots_gpu) > 0: XYPlot(plots_gpu).save(prefix + dim + '_GPU.pdf')
		if len(plots_cpu) > 0: XYPlot(plots_cpu).save(prefix + dim + '_CPU.pdf')
