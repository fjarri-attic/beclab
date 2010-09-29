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

	return vis_data, pop_data

XYPlot([test(noise=False, gpu=False, nvz=32, losses=False, wigner=False, a12=(100.4 + 98.13)/2)[0]]).save("for_m.pdf")
exit(1)
#XYPlot([
#	test(noise=True, gpu=False, a12=97.9, nvz=16, ensembles=64),
#	test(noise=True, gpu=True, a12=97.9, nvz=16, ensembles=64)
#	], location="upper right").save("1d_visibility.pdf")
#XYPlot([test(noise=False, gpu=True, a12=97.9, nvz=1024, ensembles=64)[0] for i in xrange(1)],
#	location="upper right").save("1d_visibility.pdf")
#XYPlot([test(noise=True, gpu=True, a12=97.9, nvz=32, ensembles=512)[0] for i in xrange(5)],
#	).save("test1.pdf")
#XYPlot([test(noise=True, gpu=True, a12=97.9, nvz=32, ensembles=512)[0] for i in xrange(5)],
#	legend=False).save("test2.pdf")
#XYPlot([test(noise=True, gpu=True, a12=97.9, nvz=32, ensembles=512)[0] for i in xrange(5)],
#	legend=False, gradient=True).save("test3.pdf")

#XYPlot([test(noise=False, gpu=True, a12=80.8, nvz=x)[0] for x in [16,32,64,128,256,512,1024,2048]],
#	location="lower left").save("1d_visibility.pdf")
#XYPlot([
#	test(noise=True, gpu=False, a12=97.9, nvz=16, ensembles=64)
#], location="upper right").save("1d_visibility.pdf")

#v1, p1 = test(noise=True, gpu=True, a12=97.9, nvz=32, ensembles=64, dt_evo=1e-5)
#v2, p2 = test(noise=False, gpu=True, a12=97.9, nvz=32, ensembles=64, dt_evo=1e-5)
#XYPlot([p1, p2]).save("1d_population.pdf")

# showcase 1: visibility(t) & N1(t) for different lattice sizes (no diffusion, no initial noise)
#results = [test(noise=False, wigner=False, gpu=True, nvz=nvz, a12=97.9)
#	for nvz in [16,32,64,128,256,512,1024,2048]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("1_visibility_979.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("1_population_979.pdf")
#results = [test(noise=False, wigner=False, gpu=True, nvz=nvz, a12=80.8)
#	for nvz in [16,32,64,128,256,512,1024,2048]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("1_visibility_808.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("1_population_808.pdf")

# showcase 2: visibility(t) & N1(t) for different time steps (no diffusion, no initial noise)
#results = [test(noise=False, wigner=False, gpu=True, nvz=1024, a12=97.9, dt_evo=dt)
#	for dt in [4e-5, 2e-5, 1e-5, 4e-6, 2e-6, 1e-6, 4e-7]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("2_visibility_979.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("2_population_979.pdf")
#results = [test(noise=False, wigner=False, gpu=True, nvz=1024, a12=80.8, dt_evo=dt)
#	for dt in [4e-5, 2e-5, 1e-5, 4e-6, 2e-6, 1e-6, 4e-7]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("2_visibility_808.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("2_population_808.pdf")

# showcase 3: visibility(t) & N1(t) for different lattice sizes (no diffusion, but with initial noise)
#results = [test(noise=False, wigner=True, ensembles=512, gpu=True, nvz=nvz, a12=97.9)
#	for nvz in [16,32,64,128,256]]
#XYPlot([r[0] for r in results], title="Wigner + vacuum noise, visibility", gradient=True).save("3_visibility_979.pdf")
#XYPlot([r[1] for r in results], title="Wigner + vacuum noise, N1", gradient=True).save("3_population_979.pdf")
#results = [test(noise=False, wigner=True, ensembles=512, gpu=True, nvz=nvz, a12=80.8)
#	for nvz in [16,32,64,128,256]]
#XYPlot([r[0] for r in results], title="Wigner + vacuum noise, visibility", gradient=True).save("3_visibility_808.pdf")
#XYPlot([r[1] for r in results], title="Wigner + vacuum noise, N1", gradient=True).save("3_population_808.pdf")

# showcase 4: visibility(t) & N1(t) - divergence of trajectories (no diffusion, but with initial noise)
results = [test(noise=False, wigner=True, gpu=True, nvz=32, ensembles=512, a12=97.9)[0]
	for i in xrange(10)]
XYPlot(results, legend=False, title=results[0].name).save("4_visibility_979.pdf")
results = [test(noise=False, wigner=True, gpu=True, nvz=32, ensembles=512, a12=80.8)[0]
	for i in xrange(10)]
XYPlot(results, legend=False, title=results[0].name).save("4_visibility_808.pdf")
