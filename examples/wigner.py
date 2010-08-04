import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics

def testWigner(gpu):
	constants = Constants(Model(N=30000, detuning=-41, nvx=16, nvy=16, nvz=128,
		ensembles=2), double_precision=False)
	env = Environment(gpu=gpu)
	evolution = SplitStepEvolution2(env, constants)
	pulse = Pulse(env, constants)
	a = VisibilityCollector(env, constants, verbose=True)
	b = ParticleNumberCollector(env, constants, verbose=True, pulse=pulse, matrix_pulse=True)

	gs = GPEGroundState(env, constants)

	cloud = gs.createCloud()
	cloud.toWigner()

	#evolution.run(cloud, 0.02, callbacks=[], callback_dt=1)
	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t1 = time.time()
	evolution.run(cloud, 0.4, callbacks=[a, b], callback_dt=0.01)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	#times, Na, Nb, N = b.getData()
	#XYPlot([XYData("test", times, (Na-Nb)/N, ymin=-1, ymax=1, xname="Time, s")]).save('test.pdf')

	times, vis = a.getData()
	vis = XYData("noise", times, vis, ymin=0, ymax=1,
		xname="Time, ms", yname="Visibility")
	vis = XYPlot([vis])
	vis.save('test.pdf')

testWigner(False)
testWigner(True)