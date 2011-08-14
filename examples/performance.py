import numpy
import time
import itertools

import hotshot, hotshot.stats

from beclab import *

numpy.random.seed(123)

def runTest(func, use_profiler, *args):
	env = envs.cuda(sync_calls=use_profiler)
	try:
		return func(env, use_profiler, *args)
	finally:
		env.release()

def testEvolution(env, use_profiler, test_name, use_collectors):

	if use_profiler:
		prefix = "performance_" + test_name + "_"
		prof_preparation = prefix + "prep.prof"
		prof_gs = prefix + "gs.prof"
		prof_evo = prefix + "evo.prof"

	print "=" * 80
	print "Testing {test_name} evolution {collectors} and {profiling}".format(
		test_name=test_name,
		collectors=("with collectors" if use_collectors else "without collectors"),
		profiling=("with profiling" if use_profiler else "without profiling")
	)
	print

	total_N = 50000
	total_time = 0.05
	shape = (50, 10, 10)
	ensembles = 16
	e_cut = 12000

	constants = Constants(double=env.supportsDouble(), e_cut=e_cut)

	# Preparation

	if use_profiler:
		prof = hotshot.Profile(prof_preparation)
		prof.start()
	else:
		t = time.time()

	if test_name == 'harmonic-rk':
		grid = HarmonicGrid(constants, shape)
		gs = RK5HarmonicGroundState(env, constants, grid, Nscale=total_N, eps=1e-7, components=2)
	elif test_name in ('uniform-ss', 'uniform-rk'):
		grid = UniformGrid.forN(constants, total_N, shape)
		gs = SplitStepGroundState(env, constants, grid, precision=1e-6, dt=1e-5, components=2)

	if test_name == 'harmonic-rk':
		evolution = RK5HarmonicEvolution(env, constants, grid,
			atol_coeff=1e-3, eps=1e-6, Nscale=total_N, components=2, ensembles=ensembles)
	elif test_name == 'uniform-ss':
		evolution = SplitStepEvolution(env, constants, grid,
			dt=1e-5, components=2, ensembles=ensembles)
	elif test_name == 'uniform-rk':
		evolution = RK5IPEvolution(env, constants, grid,
			atol_coeff=1e-3, eps=1e-6, Nscale=total_N, components=2, ensembles=ensembles)

	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350,
		components=2, ensembles=ensembles)

	if use_collectors:
		p = ParticleNumberCollector(env, constants, grid, pulse=pulse)
		p.prepare(components=2, ensembles=ensembles)
		v = VisibilityCollector(env, constants, grid)
		v.prepare(components=2, ensembles=ensembles)
		evo_kwds = dict(callbacks=[p, v], callback_dt=total_time / 100)
	else:
		evo_kwds = {}

	if use_profiler:
		prof.stop()
		prof.close()
	else:
		t_prep = time.time() - t

	# GS creation

	if use_profiler:
		prof = hotshot.Profile(prof_gs)
		prof.start()
	else:
		t = time.time()

	psi = gs.create((total_N, 0))

	if use_profiler:
		prof.stop()
		prof.close()
	else:
		t_gs = time.time() - t

	psi.toWigner(ensembles)
	pulse.apply(psi, theta=0.5 * numpy.pi)

	# Evolution

	if use_profiler:
		prof = hotshot.Profile(prof_evo)
		prof.start()
	else:
		t = time.time()

	evolution.run(psi, total_time, **evo_kwds)
	env.synchronize()

	if use_profiler:
		prof.stop()
		prof.close()
	else:
		t_evo = time.time() - t

	if use_profiler:
		for prof_name in (prof_preparation, prof_gs, prof_evo):
			print "-" * 80
			print prof_name
			print

			stats = hotshot.stats.load(prof_name)
			stats.strip_dirs()
			stats.sort_stats('cumulative', 'calls')
			stats.print_stats(20)
	else:
		print ("Preparation: {t_prep:.3f} s\n" +
			"GS creation: {t_gs:.3f} s,\n" +
			"Evolution: {t_evo:.3f} s").format(t_prep=t_prep, t_gs=t_gs, t_evo=t_evo)
	print


if __name__ == '__main__':

	tests = (
		('harmonic-rk', 'uniform-ss', 'uniform-rk',),
		(False, True,), # use collectors
		(False, True,), # use profiler
	)

	for test_name, use_collectors, use_profiler in itertools.product(*tests):
		runTest(testEvolution, use_profiler, test_name, use_collectors)
