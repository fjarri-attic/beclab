import numpy
import time
import math

from beclab import *

def testAxial(gpu, matrix_pulses):
	# preparation
	env = Environment(gpu=gpu)
	constants = Constants(Model(N=150000, detuning=-41),
		double_precision=False if gpu else True)

	gs = GPEGroundState(env, constants)
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = AxialProjectionCollector(env, constants, matrix_pulse=matrix_pulses, pulse=pulse)

	# experiment
	cloud = gs.createCloud()

	pulse.apply(cloud, theta=0.5 * math.pi, matrix=matrix_pulses)

	t1 = time.time()
	evolution.run(cloud, time=0.1, callbacks=[a], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, picture = a.getData()

	return HeightmapPlot(
		HeightmapData("test", picture,
			xmin=0, xmax=100,
			ymin=-constants.zmax * constants.l_rho * 1e6,
			ymax=constants.zmax * constants.l_rho * 1e6,
			zmin=-1, zmax=1,
			xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
	)

for gpu, matrix_pulses in ((False, True), (False, False), (True, True), (True, False)):
	suffix = ("gpu" if gpu else "cpu") + "_" + ("ideal" if matrix_pulses else "nonideal") + "_pulses"
	testAxial(gpu=gpu, matrix_pulses=matrix_pulses).save("axial_" + suffix + ".pdf")
