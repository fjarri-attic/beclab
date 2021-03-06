import numpy
import time
import itertools

from beclab import *
import gc


def testConvergence(grid_type, dim, prop_type, repr_type):
	env = envs.cuda()
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
		'3d': 150000
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
	ss_dts = {
		'1d': [4e-5, 2e-5, 1e-5, 4e-6, 2e-6],
		'3d': [1e-4, 4e-5, 2e-5, 1e-5, 4e-6]
	}[dim]

	# relative tolerances for RK5
	rk_epss = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

	ensembles = 128

	wigner = (repr_type == 'wigner')

	constants = Constants(double=env.supportsDouble(), **constants_kwds)
	grid = UniformGrid.forN(env, constants, total_N, shape)

	gs = SplitStepGroundState(env, constants, grid, dt=ss_gs_dt)
	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350)

	# experiment
	final_vis = []
	final_N = []
	vis_plots = []

	if prop_type == 'split-step':
		params = ss_dts
		pname = 'dt'
	elif prop_type == 'rk5':
		params = rk_epss
		pname = 'eps'

	for param in params:
		psi = gs.create((total_N, 0))

		if wigner:
			psi.toWigner(ensembles)

		p = ParticleNumberCollector(env, constants, grid, pulse=pulse)
		v = VisibilityCollector(env, constants, grid)

		if prop_type == 'split-step':
			evolution = SplitStepEvolution(env, constants, grid, dt=param)
		elif prop_type == 'rk5':
			evolution = RK5IPEvolution(env, constants, grid, Nscale=total_N,
				atol_coeff=1e-3, eps=param)

		pulse.apply(psi, theta=0.5 * numpy.pi)

		evolution.run(psi, 1.0, callbacks=[p, v], callback_dt=0.005)
		env.synchronize()

		times, vis = v.getData()
		times, Ns, Ntotals = p.getData()

		final_vis.append(vis[-1])
		final_N.append(Ntotals[-1])

		print "  {pname} = {param}: N = {N:.4f}, V = {V:.4f}".format(
			pname=pname, param=param, N=final_N[-1], V=final_vis[-1])

		vis_plots.append(XYData(
			pname + " = " + str(param), times * 1e3, vis,
			xname="T (ms)", yname="$\\mathcal{V}$", ymin=0, ymax=1
		))

		del p, v, evolution
		gc.collect()

	return vis_plots


if __name__ == '__main__':

	prefix = 'convergence_'

	tests = (
		('uniform',), # grid type
		('split-step', 'rk5',), # propagation type
		('classical', 'wigner',), # representation type
	)

	for dim in ('1d', '3d',):
		print "\n*** {dim} ***\n".format(dim=dim)

		for grid_type, prop_type, repr_type in itertools.product(*tests):

			if grid_type == 'harmonic' and prop_type == 'split-step':
				continue

			print "* Testing", ", ".join((grid_type, prop_type, repr_type))
			plots = testConvergence(grid_type, dim, prop_type, repr_type)
			XYPlot(plots).save(prefix + dim + '_' + grid_type + '_' +
				prop_type + '_' + repr_type + '.pdf')
