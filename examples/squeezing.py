import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics

def testUncertainties(gpu):

	# normal parameters
	m = Model(N=40000, detuning=-41, nvx=8, nvy=8, nvz=64, ensembles=4, e_cut=1e6)
	t = 1.0
	callback_dt = 0.002
	noise = True

	# Yun Li, shallow trap, no losses
	#m = Model(N=20000, nvx=16, nvy=16, nvz=16, ensembles=4, e_cut=1e6,
	#	a11=100.44, a12=88.28, a22=95.47, fx=42.6, fy=42.6, fz=42.6,
	#	gamma111=0, gamma12=0, gamma22=0
	#	)
	#t = 0.5
	#callback_dt = 0.001
	#noise = False

	# Yun Li, steep trap, no losses
	#m = Model(N=100000, nvx=16, nvy=16, nvz=16, ensembles=4, e_cut=1e6,
	#	a11=100.44, a12=88.28, a22=95.47, fx=2e3, fy=2e3, fz=2e3, dt_evo=1e-7, dt_steady=1e-7,
	#	gamma111=0, gamma12=0, gamma22=0
	#	)
	#t = 0.005
	#callback_dt = 0.00005
	#noise = False

	constants = Constants(m, double=False if gpu else True)
	env = envs.cuda() if gpu else envs.cpu()
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = VisibilityCollector(env, constants, verbose=True)
	b = ParticleNumberCollector(env, constants, verbose=True, pulse=pulse, matrix_pulse=True)
	u = UncertaintyCollector(env, constants)

	gs = GPEGroundState(env, constants)

	cloud = gs.createCloud()
	cloud.toWigner()

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t1 = time.time()
	evolution.run(cloud, t, callbacks=[a, u], callback_dt=callback_dt, noise=noise)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, na, nb, sp = u.getData()
	XYPlot([
		XYData("|1>", times * 1000, na, ymin=0, xname="Time, ms", yname="$\\Delta$N"),
		XYData("|2>", times * 1000, nb, ymin=0, xname="Time, ms", yname="$\\Delta$N")
	]).save(	'N_stddev' + str(env) + '.pdf')

	XYPlot([XYData("test", times * 1000, numpy.log10(sp), xname="Time, ms", yname="log10($\\xi^2$)")]).save(
		"XiSquared_" + str(env) + ".pdf")

	env.release()

testUncertainties(True)
#testPhaseNoise(False)