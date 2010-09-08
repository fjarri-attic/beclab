import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics

constants = Constants(Model(N=60, nvx=1, nvy=1, nvz=32, ensembles=16,
	fx=42e3, fy=42e3, fz=90, dt_evo=1e-5, border=2.0, e_cut=1,
	a11=100.4, a12=97.9, a22=95.5, detuning=0,
	gamma111=0, gamma12=0, gamma22=0), double_precision=True)
env = Environment(gpu=False)

evolution = SplitStepEvolution2(env, constants)

gs = GPEGroundState(env, constants)
pulse = Pulse(env, constants)
v = VisibilityCollector(env, constants, verbose=False)
p = ParticleNumberCollector(env, constants, pulse=pulse, matrix_pulse=True, verbose=True)
a = AxialProjectionCollector(env, constants, matrix_pulse=True, pulse=pulse)
ps = ParticleStatistics(env, constants)

cloud = gs.createCloud()
cloud.toWigner()

#evolution.run(cloud, 0.05)
pulse.apply(cloud, math.pi * 0.5, matrix=True)

t1 = time.time()
evolution.run(cloud, 0.05, callbacks=[v, p, a], callback_dt=0.0005)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, vis = v.getData()
XYPlot([XYData("test", times, vis, ymin=0, ymax=1,
	xname="Time, s", yname="Visibility")]).save("1d_visibilty.pdf")

times, N1, N2, N = p.getData()
XYPlot([XYData("test", times, N1,
	ymin=0, ymax=60, xname="Time, s", yname="Population, N1")]).save("1d_population.pdf")

times, picture = a.getData()
HeightmapPlot(HeightmapData("test", picture,
	xmin=0, xmax=50,
	ymin=-constants.zmax * 1e6,
	ymax=constants.zmax * 1e6,
	#zmin=-1, zmax=1,
	zmin=0,
	xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
).save("1d_axial.pdf")
