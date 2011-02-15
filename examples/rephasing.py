import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics

def runPass(evolution, pulse, cloud, t):
	pulse.apply(cloud, math.pi, matrix=True)
	evolution.run(cloud, t, noise=False)

def testRephasing(gpu):

	m = Model(N=44000, detuning=-41, nvx=8, nvy=8, nvz=64, ensembles=1, e_cut=1e6, dt_evo=1e-5)
	t_max = 0.5
	t_step = 0.05

	constants = Constants(m, double=False if gpu else True)
	env = envs.cuda() if gpu else envs.cpu()
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	stats = ParticleStatistics(env, constants)

	gs = GPEGroundState(env, constants)

	times = [0.0]
	vis = [1.0]

	cloud = gs.createCloud()
	#cloud.toWigner()

	pulse.apply(cloud, 0.5 * math.pi, matrix=True)

	t = t_step
	while t <= t_max:
		evolution.run(cloud, t_step / 2, noise=False)
		print "Pre-pi step finished, t=" + str(cloud.time)
		new_cloud = cloud.copy()
		pulse.apply(new_cloud, math.pi, matrix=True)
		evolution.run(new_cloud, t / 2, noise=False)
		print "Post-pi step finished"

		times.append(t)
		vis.append(stats.getVisibility(new_cloud.a, new_cloud.b))
		print "Visibility=" + str(vis[-1])

		del new_cloud
		t += t_step

	times = numpy.array(times)
	vis = numpy.array(vis)
	XYPlot([XYData("test", times, vis, ymin=0, ymax=1, xname="Time, s", yname="Visibility")]).save(
		"rephasing_" + str(env) + ".pdf")

	env.release()

testRephasing(True)
#testRephasing(False)