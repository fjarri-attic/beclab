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
		double=(not gpu))
	env = envs.cuda() if gpu else envs.cpu()

	evolution = SplitStepEvolution(env, constants)

	gs = GPEGroundState(env, constants)
	pulse = Pulse(env, constants)
	v = VisibilityCollector(env, constants, verbose=False)
	p = ParticleNumberCollector(env, constants, pulse=pulse, matrix_pulse=True, verbose=False)

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

	env.release()

	return vis_data, pop_data

# showcase 1: visibility(t) & N1(t) for different lattice sizes (no diffusion, no initial noise)
#results = [test(noise=False, wigner=False, gpu=True, nvz=nvz, a12=97.9)
#	for nvz in [16,32,64,128,256,512,1024,2048]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("visibility_1d_s1_979.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("1_population_979.pdf")
#results = [test(noise=False, wigner=False, gpu=True, nvz=nvz, a12=80.8)
#	for nvz in [16,32,64,128,256,512,1024,2048]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("visibility_1d_s1_808.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("1_population_808.pdf")

# showcase 2: visibility(t) & N1(t) for different time steps (no diffusion, no initial noise)
#results = [test(noise=False, wigner=False, gpu=True, nvz=1024, a12=97.9, dt_evo=dt)
#	for dt in [4e-5, 2e-5, 1e-5, 4e-6, 2e-6, 1e-6, 4e-7]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("visibility_1d_s2_979.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("2_population_979.pdf")
#results = [test(noise=False, wigner=False, gpu=True, nvz=1024, a12=80.8, dt_evo=dt)
#	for dt in [4e-5, 2e-5, 1e-5, 4e-6, 2e-6, 1e-6, 4e-7]]
#XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save("visibility_1d_s2_808.pdf")
#XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save("2_population_808.pdf")

# showcase 3: visibility(t) & N1(t) for different lattice sizes (no diffusion, but with initial noise)
#results = [test(noise=False, wigner=True, ensembles=512, gpu=True, nvz=nvz, a12=97.9)
#	for nvz in [16,32,64,128,256]]
#XYPlot([r[0] for r in results], title="Wigner + vacuum noise, visibility", gradient=True).save("visibility_1d_s3_979.pdf")
#XYPlot([r[1] for r in results], title="Wigner + vacuum noise, N1", gradient=True).save("visibility_1d_s3_979.pdf")
#results = [test(noise=False, wigner=True, ensembles=512, gpu=True, nvz=nvz, a12=80.8)
#	for nvz in [16,32,64,128,256]]
#XYPlot([r[0] for r in results], title="Wigner + vacuum noise, visibility", gradient=True).save("visibility_1d_s3_808.pdf")
#XYPlot([r[1] for r in results], title="Wigner + vacuum noise, N1", gradient=True).save("visibility_1d_s3_808.pdf")

# showcase 4: visibility(t) & N1(t) - divergence of trajectories (no diffusion, but with initial noise)
results = [test(noise=False, wigner=True, gpu=True, nvz=32, ensembles=512, a12=97.9)[0]
	for i in xrange(10)]
XYPlot(results, legend=False, title=results[0].name).save("visibility_1d_s4_979.pdf")
results = [test(noise=False, wigner=True, gpu=True, nvz=32, ensembles=512, a12=80.8)[0]
	for i in xrange(10)]
XYPlot(results, legend=False, title=results[0].name).save("visibility_1d_s4_808.pdf")
