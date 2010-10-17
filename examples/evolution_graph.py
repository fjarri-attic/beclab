import numpy
import time
import math

from beclab import *

def testEvolutionGraph(gpu):
	# preparation
	env = envs.cuda() if gpu else envs.cpu()
	constants = Constants(Model(N=150000), double=False if gpu else True)
	gs = GPEGroundState(env, constants)
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = SurfaceProjectionCollector(env, constants, pulse=pulse)

	# experiment
	cloud = gs.createCloud()
	pulse.apply(cloud, theta=0.5*math.pi)
	t1 = time.time()
	evolution.run(cloud, time=0.119, callbacks=[a], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	# render
	times, a_xy, a_yz, b_xy, b_yz = a.getData()

	times = [str(int(x * 1000 + 0.5)) for x in times]

	def constructPlot(dataset, dl):
		hms = []

		# good approximation for the scale: density in the middle of TF ground state
		max_density = constants.muTF() / constants.g[(COMP_1_minus1, COMP_1_minus1)] * dl

		for t, hm in zip(times, dataset):
			hms.append(HeightmapData(t, hm.transpose(),
				xmin=-constants.zmax, xmax=constants.zmax,
				ymin=-constants.ymax, ymax=constants.ymax,
				zmin=0, zmax=max_density))

		return EvolutionPlot(hms, shape=(6, 4))

	res = constructPlot(a_yz, constants.xmax), constructPlot(b_yz, constants.xmax)
	env.release()
	return res

p1, p2 = testEvolutionGraph(gpu=False)
p1.save("evolution_cpu_a.pdf")
p2.save("evolution_cpu_b.pdf")

p1, p2 = testEvolutionGraph(gpu=True)
p1.save("evolution_gpu_a.pdf")
p2.save("evolution_gpu_b.pdf")