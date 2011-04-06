import numpy
import time
import math

from beclab import *
from beclab.evolution import SplitStepEvolution2

def testVisibility(wigner, gpu):

	t = 1e-3
	dt_evo = t / 1000.0
	rabi_freq = 350
	detuning = -37
	callback_dt = t / 100.0

	constants = Constants(Model(N=44000, nvx=8, nvy=8, nvz=64, e_cut=1e6, dt_evo=dt_evo, ensembles=4),
		double=True)
	env = envs.cuda()

	evolution = SplitStepEvolution2(env, constants, rabi_freq=rabi_freq, detuning=detuning)
	pulse = Pulse(env, constants)

	gs = GPEGroundState(env, constants)
	v = VisibilityCollector(env, constants, verbose=True)
	p = ParticleNumberCollector(env, constants, pulse=None, matrix_pulse=True, verbose=True)

	cloud = gs.createCloud()
	if wigner:
		cloud.toWigner()

	t1 = time.time()
	evolution.run(cloud, t, callbacks=[v, p], callback_dt=callback_dt, noise=wigner)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	name = str(env) + ", " + ("Wigner" if wigner else "GPE")

	times, vis = v.getData()
	vis = XYData(name, times, vis, ymin=0, ymax=1, xname="Time, s", yname="Visibility")

	times, N1, N2, N = p.getData()
	particles = XYData(name, times, (N1 - N2) / N,
		ymin=-1, ymax=1, xname="Time, s", yname="Population ratio")

	env.release()
	return particles, vis

visibility_data = []
particles_data = []
for wigner, gpu in ((False, False), (False, True), (True, False), (True, True)):
	p, v = testVisibility(wigner, gpu)
	visibility_data.append(v)
	particles_data.append(p)

XYPlot(visibility_data).save('coupling_vis.pdf')
XYPlot(particles_data).save('coupling_population.pdf')