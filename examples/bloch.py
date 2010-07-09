import numpy
import time
import math

from beclab import *

constants = Constants(Model())
env = Environment(gpu=False)

gs = GPEGroundState(env, constants)
evolution = SplitStepEvolution(env, constants)

experiments_num = 64
bas = [BlochSphereAveragesCollector(env) for i in xrange(experiments_num)]

for i in xrange(experiments_num):
	cloud = gs.createCloud()
	pulse = Pulse(env, constants, starting_phase=numpy.random.normal(scale=1.0/math.sqrt(env.constants.N)))
	pulse.apply(cloud, theta=0.5 * math.pi + numpy.random.normal(scale=1.0/math.sqrt(env.constants.N)))

	t1 = time.time()
	evolution.run(cloud, 0.01, [bas[i]], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

snapshots = BlochSphereAveragesCollector.getSnapshots(bas)
times = bas[0].times

for t, snapshot in zip(times, snapshots):
	pr = HeightmapData("BS projection test", snapshot, xmin=0, xmax=2 * math.pi,
		ymin=0, ymax=math.pi, zmin=0, xname="Phase", yname="Amplitude", zname="Density")
	pr = HeightmapPlot(pr)
	pr.save('test' + str(int(t * 1000 + 0.5)).zfill(3) + '.png')
