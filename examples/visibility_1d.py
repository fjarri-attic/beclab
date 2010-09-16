import numpy
import time
import math

from beclab import *
from beclab.state import ParticleStatistics


def test(gpu=False, ensembles=256, nvz=32, dt_evo=1e-5, a12=97.9,
	losses=True, equilibration_time=0, noise=True, wigner=True,
	zero_gs=False, e_cut=1):

	kwds = {}

	if not losses:
		kwds['gamma111'] = 0
		kwds['gamma12'] = 0
		kwds['gamma22'] = 0

	constants = Constants(Model(N=60, nvx=1, nvy=1, nvz=nvz, ensembles=ensembles,
		fx=42e3, fy=42e3, fz=90, dt_evo=dt_evo, border=2.0, e_cut=e_cut,
		a11=100.4, a12=a12, a22=95.5, detuning=0, **kwds),
		double_precision=(not gpu))
	env = Environment(gpu=gpu)

	evolution = SplitStepEvolution(env, constants)

	gs = GPEGroundState(env, constants)
	pulse = Pulse(env, constants)
	v = VisibilityCollector(env, constants, verbose=False)
	p = ParticleNumberCollector(env, constants, pulse=pulse, matrix_pulse=True, verbose=False)
	#a = AxialProjectionCollector(env, constants, matrix_pulse=True, pulse=pulse)

	#ps = ParticleStatistics(env, constants)

	cloud = gs.createCloud()
	if zero_gs:
		cloud.a._initializeMemory()
	if wigner:
		cloud.toWigner()

	if equilibration_time > 0:
		evolution.run(cloud, equilibration_time, noise=noise)

	pulse.apply(cloud, math.pi * 0.5, matrix=True)

	t1 = time.time()
	evolution.run(cloud, 0.05, callbacks=[v, p], callback_dt=0.001, noise=noise)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	name = ["gpu" if gpu else "cpu"] + \
		([str(ensembles) + " ens."] if wigner else []) + \
		[str(nvz) + " cells",
		"dt = " + str(dt_evo * 1e6) + " $\mu$s",
		"a12 = " + str(a12)] + \
		([] if losses else ["no losses"]) + \
		(["eq. " + str(equilibration_time * 1e3) + " ms"] if equilibration_time > 0 else []) + \
		(["noise"] if noise else [])

	name = ", ".join(name)

	times, vis = v.getData()
	vis_data = XYData(name, times * 1e3, vis, ymin=0, ymax=1,
		xname="Time, ms", yname="Visibility")

	times, N1, N2, N = p.getData()
	pop_data = XYData(name, times * 1e3, N1,
		ymin=0, ymax=60, xname="Time, ms", yname="Population, N1")

	#times, picture = a.getData()
	#axial_data = HeightmapData(name, picture,
	#	xmin=0, xmax=50,
	#	ymin=-constants.zmax * 1e6,
	#	ymax=constants.zmax * 1e6,
	#	zmin=0, zmax=1,
	#	xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")

	return vis_data

#v1 = test(noise=False, gpu=True)
#v2 = test(noise=False, gpu=False)

XYPlot([test(noise=False, gpu=True, a12=80.8) for i in xrange(10)], location="upper right").save("1d_visibility.pdf")