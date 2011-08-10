import numpy
import time
import itertools

from beclab import *
from beclab.meters import ParticleStatistics


def testAxialView(gpu, matrix_pulses, grid_type, dim, prop_type, use_cutoff):
	env = envs.cuda() if gpu else envs.cpu()
	try:
		return runTest(env, matrix_pulses, grid_type, dim, prop_type, use_cutoff)
	finally:
		env.release()

def runTest(env, matrix_pulses, grid_type, dim, prop_type, use_cutoff):

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
		('3d', 'uniform'): (128, 8, 8),
		('1d', 'harmonic'): (50,),
		('3d', 'harmonic'): (50, 10, 10)
	}[(dim, grid_type)]

	# time step for split-step propagation
	ss_dt = {
		'1d': 1e-6,
		'3d': 1e-5
	}[dim]

	e_cut = {
		'1d': 12000,
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

	if grid_type == 'uniform':
		gs = SplitStepGroundState(env, constants, grid, dt=ss_dt)
	elif grid_type == 'harmonic':
		gs = RK5HarmonicGroundState(env, constants, grid, eps=1e-7)

	if grid_type == 'harmonic':
		evolution = RK5HarmonicEvolution(env, constants, grid,
			atol_coeff=1e-3, eps=1e-6, Nscale=total_N)
	elif prop_type == 'split-step':
		evolution = SplitStepEvolution(env, constants, grid, dt=ss_dt)
	elif prop_type == 'rk5':
		evolution = RK5IPEvolution(env, constants, grid, Nscale=total_N,
			atol_coeff=1e-3, eps=1e-6)

	if matrix_pulses:
		pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350)
	else:
		if grid_type == 'harmonic':
			pulse = EvolutionPulse(env, constants, grid, RK5HarmonicEvolution,
				f_detuning=41, f_rabi=350, Nscale=total_N)
		elif prop_type == 'split-step':
			pulse = EvolutionPulse(env, constants, grid, SplitStepEvolution,
				f_detuning=41, f_rabi=350, dt=1e-6)
		elif prop_type == 'rk5':
			pulse = EvolutionPulse(env, constants, grid, RK5IPEvolution,
				Nscale=total_N, f_detuning=41, f_rabi=350,
				atol_coeff=1e-3, eps=1e-6)

	a = AxialProjectionCollector(env, constants, grid, pulse=pulse)
	p = ParticleNumberCollector(env, constants, grid, pulse=pulse, verbose=True)
	v = VisibilityCollector(env, constants, grid, verbose=True)

	# experiment
	psi = gs.create((total_N, 0))

	pulse.apply(psi, theta=0.5 * numpy.pi)

	t1 = time.time()
	evolution.run(psi, 0.1, callbacks=[a, p, v], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()

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

	print ("  Final N: {N} ({Ntotal}), V: {vis}\n" +
		"  Time spent: {t} s").format(N=Ns[:,-1], Ntotal=Ntotals[-1], vis=vis[-1], t=t2-t1)

	return res


if __name__ == '__main__':

	prefix = 'axial_view_'

	tests = (
		('uniform', 'harmonic',), # grid type
		('split-step', 'rk5',), # propagation type
		(False, True,), # matrix pulses
		(False, True,), # cutoff usage
		(False, True,), # gpu usage
	)

	for dim in ('1d', '3d',):
		print "\n*** {dim} ***\n".format(dim=dim)

		for grid_type, prop_type, matrix_pulses, use_cutoff, gpu in itertools.product(*tests):

			if grid_type == 'harmonic' and prop_type == 'split-step':
				continue

			print "* Testing", ", ".join((
				grid_type, prop_type,
				("matrix pulses" if matrix_pulses else "real pulses"),
				("cutoff" if use_cutoff else "no cutoff"),
				("GPU" if gpu else "CPU")))

			testAxialView(gpu, matrix_pulses, grid_type, dim, prop_type, use_cutoff).save(
				prefix + dim + '_' + grid_type + '_' + prop_type +
				('_matrix' if matrix_pulses else '_real') +
				('_cutoff' if use_cutoff else '_nocutoff') +
				('_GPU' if gpu else '_CPU') + '.pdf')
