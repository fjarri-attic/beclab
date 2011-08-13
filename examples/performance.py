import numpy
import time

import hotshot, hotshot.stats

from beclab import *

numpy.random.seed(123)

def runTest(*args):
	env = envs.cpu()
	try:
		return testHarmonicWigner3D(env, *args)
	finally:
		env.release()

def testHarmonicWigner3D(env):

	prof_preparation = "performance_wigner_prep.prof"
	prof_gs = "performance_wigner_gs.prof"
	prof_evo = "performance_wigner_evo.prof"

	total_N = 50000
	total_time = 0.05
	shape = (50, 10, 10)
	ensembles = 64
	e_cut = 12000

	prof = hotshot.Profile(prof_preparation)
	prof.start()

	constants = Constants(double=env.supportsDouble(), e_cut=e_cut)

	grid = HarmonicGrid(constants, shape)
	gs = RK5HarmonicGroundState(env, constants, grid, eps=1e-7,
		components=2)
	evolution = RK5HarmonicEvolution(env, constants, grid,
			atol_coeff=1e-3, eps=1e-6, Nscale=total_N, components=2, ensembles=ensembles)
	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350,
		components=2, ensembles=ensembles)

	p = ParticleNumberCollector(env, constants, grid, pulse=pulse)
	p.prepare(components=2, ensembles=ensembles)
	v = VisibilityCollector(env, constants, grid)
	v.prepare(components=2, ensembles=ensembles)

	prof.stop()
	prof.close()

	# experiment
	prof = hotshot.Profile(prof_gs)
	prof.start()
	psi = gs.create((total_N, 0))
	prof.stop()
	prof.close()

	psi.toWigner(ensembles)

	pulse.apply(psi, theta=0.5 * numpy.pi)

	prof = hotshot.Profile(prof_evo)
	prof.start()
	evolution.run(psi, total_time, callbacks=[p, v], callback_dt=total_time / 100)
	env.synchronize()
	prof.stop()
	prof.close()

	for prof_name in (prof_preparation, prof_gs, prof_evo):
		print "=" * 80
		print prof_name
		print "=" * 80
		print

		stats = hotshot.stats.load(prof_name)
		stats.strip_dirs()
		stats.sort_stats('cumulative', 'calls')
		stats.print_stats(20)

if __name__ == '__main__':
	runTest()
