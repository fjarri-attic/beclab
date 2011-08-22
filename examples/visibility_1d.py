"""
Visibility for Ramsey sequence in 1D case and different parameters
"""

import numpy
from beclab import *

def test(ensembles=256, points=40, a12=97.9,
	losses=True, equilibration_time=0, wigner=True,
	e_cut=3000, eps=1e-6):

	N = 60
	parameters = dict(use_effective_area=True,
		fx=42e3, fy=42e3, fz=90,
		a11=100.4, a12=a12, a22=95.5)

	if not losses:
		parameters.update(dict(gamma111=0, gamma12=0, gamma22=0))

	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(),
		e_cut=e_cut, **parameters)
	grid = HarmonicGrid(constants, (points,))

	gs = RK5HarmonicGroundState(env, constants, grid, Nscale=N)
	evolution = RK5HarmonicEvolution(env, constants, grid, eps=eps, Nscale=N)
	pulse = Pulse(env, constants, grid, f_rabi=350)

	v = VisibilityCollector(env, constants, grid)
	p = ParticleNumberCollector(env, constants, grid, pulse=pulse)

	psi = gs.create((N, 0))
	if wigner:
		psi.toWigner(ensembles)

	if equilibration_time > 0:
		evolution.run(psi, equilibration_time)

	pulse.apply(psi, math.pi / 2)

	evolution.run(psi, 0.05, callbacks=[v, p], callback_dt=0.001)
	env.release()

	name = ([str(ensembles) + " ens."] if wigner else []) + \
		[str(points) + " cells",
		"eps = " + str(eps),
		"a12 = " + str(a12)] + \
		([] if losses else ["no losses"]) + \
		(["eq. " + str(equilibration_time * 1e3) + " ms"] if equilibration_time > 0 else [])

	name = ", ".join(name)

	times, vis = v.getData()
	vis_data = XYData(name, times * 1e3, vis, ymin=0, ymax=1,
		xname="T (ms)", yname="$\\mathcal{V}$")

	times, Ns, Ntotal = p.getData()
	pop_data = XYData(name, times * 1e3, Ns[0],
		ymin=0, ymax=N, xname="T (ms)", yname="Population, |1>")

	return vis_data, pop_data

if __name__ == '__main__':

	prefix = 'evolution_1d'

	# showcase 1: visibility(t) & N1(t) for different lattice sizes (no diffusion, no initial noise)
	results = [test(losses=False, wigner=False, points=points, a12=97.9)
		for points in [20, 30, 40, 50]]
	XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save(
		prefix + "_s1_V_97.9.pdf")
	XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save(
		prefix + "_s1_N_97.9.pdf")
	results = [test(losses=False, wigner=False, points=points, a12=80.8)
		for points in [20, 30, 40, 50]]
	XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save(
		prefix + "_s1_V_80.8.pdf")
	XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save(
		prefix + "_s1_N_80.8.pdf")

	# showcase 2: visibility(t) & N1(t) for different time steps (no diffusion, no initial noise)
	results = [test(losses=False, wigner=False, points=40, a12=97.9, eps=eps)
		for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]]
	XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save(
		prefix + "_s2_V_97.9.pdf")
	XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save(
		prefix + "_s2_N_97.9.pdf")
	results = [test(losses=False, wigner=False, points=40, a12=80.8, eps=eps)
		for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]]
	XYPlot([r[0] for r in results], title="GPEs, visibility", gradient=True).save(
		prefix + "_s2_V_80.8.pdf")
	XYPlot([r[1] for r in results], title="GPEs, N1", gradient=True).save(
		prefix + "_s2_N_80.8.pdf")

	# showcase 3: visibility(t) & N1(t) for different lattice sizes (no diffusion, but with initial noise)
	results = [test(losses=False, wigner=True, ensembles=512, points=points, a12=97.9)
		for points in [20, 30, 40, 50]]
	XYPlot([r[0] for r in results], title="Wigner + vacuum noise, visibility", gradient=True).save(
		prefix + "_s3_V_97.9.pdf")
	XYPlot([r[1] for r in results], title="Wigner + vacuum noise, N1", gradient=True).save(
		prefix + "_s3_N_97.9.pdf")
	results = [test(losses=False, wigner=True, ensembles=512, points=points, a12=80.8)
		for points in [20, 30, 40, 50]]
	XYPlot([r[0] for r in results], title="Wigner + vacuum noise, visibility", gradient=True).save(
		prefix + "_s3_V_80.8.pdf")
	XYPlot([r[1] for r in results], title="Wigner + vacuum noise, N1", gradient=True).save(
		prefix + "_s3_N_80.8.pdf")

	# showcase 4: visibility(t) & N1(t) - divergence of trajectories
	results = [test(losses=False, wigner=True, points=40, ensembles=512, a12=97.9)[0]
		for i in xrange(10)]
	XYPlot(results, legend=False, title=results[0].name).save(prefix + "_s4_97.9.pdf")
	results = [test(losses=False, wigner=True, points=40, ensembles=512, a12=80.8)[0]
		for i in xrange(10)]
	XYPlot(results, legend=False, title=results[0].name).save(prefix + "_s4_80.8.pdf")
