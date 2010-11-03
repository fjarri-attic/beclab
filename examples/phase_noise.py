import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics

def testPhaseNoise(gpu):
	constants = Constants(Model(N=40000, detuning=-41, nvx=8, nvy=8, nvz=64,
		ensembles=16, e_cut=1e6), double=(False if gpu else True))
	env = envs.cuda() if gpu else envs.cpu()
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = VisibilityCollector(env, constants, verbose=True)
	b = ParticleNumberCollector(env, constants, verbose=True, pulse=pulse, matrix_pulse=True)
	p = ParticleStatistics(env, constants)
	n1 = PhaseNoiseCollector(env, constants, verbose=True)
	n2 = PzNoiseCollector(env, constants, verbose=True)

	gs = GPEGroundState(env, constants)

	cloud = gs.createCloud()
	cloud.toWigner()

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t1 = time.time()
	evolution.run(cloud, 0.05, callbacks=[n1, n2, a], callback_dt=0.01, noise=False)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, vis = a.getData()
	vis = XYData("noise", times, vis, ymin=0, ymax=1,
		xname="Time, ms", yname="Visibility")
	vis = XYPlot([vis])
	vis.save('phase_noise_visibility_' + str(env) + '.pdf')

	times, noise = n1.getData()
	XYPlot([XYData("test", times * 1000, noise, ymin=0, xname="Time, ms")]).save(
		'phase_noise_' + str(env) + '.pdf')

	env.release()

testPhaseNoise(True)
testPhaseNoise(False)