import numpy
from beclab import *


parameters = [
	# Yun Li et al., steep trap, no losses
	('steep', dict(N=100000, f_trap=2e3, time=0.005, losses=False)),

	# Yun Li et al., shallow trap, no losses
	('shallow', dict(N=20000, f_trap=42.6, time=0.5, losses=False)),
]

ensembles = 128
prefix = "squeezing_"


def testSqueezing(name, **kwds):

	parameters = dict(fx=kwds['f_trap'], fy=kwds['f_trap'], fz=kwds['f_trap'],
		a11=100.44, a12=88.28, a22=95.47)
	if not kwds['losses']:
		parameters.update(dict(gamma111=0, gamma12=0, gamma22=0))
	N = kwds['N']

	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(), **parameters)
	grid = UniformGrid.forN(constants, N, (16, 16, 16))

	gs = RK5IPGroundState(env, constants, grid)
	evolution = RK5IPEvolution(env, constants, grid)
	pulse = Pulse(env, constants, grid, f_rabi=350)

	u = UncertaintyCollector(env, constants, grid)
	p = ParticleNumberCollector(env, constants, grid, verbose=True)

	psi = gs.create((N, 0))
	psi.toWigner(ensembles)
	pulse.apply(psi, numpy.pi / 2)
	evolution.run(psi, kwds['time'], callbacks=[u, p],
		callback_dt=kwds['time'] / 100)
	env.release()

	times, n_stddev, xi_squared = u.getData()

	XYPlot([
		XYData("|1>", times * 1000, n_stddev[0], ymin=0, xname="T (ms)", yname="$\\Delta$N"),
		XYData("|2>", times * 1000, n_stddev[1], ymin=0, xname="T (ms)", yname="$\\Delta$N")
	]).save(prefix + name + '_N_stddev.pdf')

	XYPlot([
		XYData("Squeezing", times * 1000, numpy.log10(xi_squared),
			xname="T (ms)", yname="log$_{10}$($\\xi^2$)")
	]).save(prefix + name + '_xi_squared.pdf')


if __name__ == '__main__':
	for name, params in parameters:
		testSqueezing(name, **params)
