"""
Check squeezing for different a12 values near Feschbach resonance.
Data is accumulated over several iterations which helps if many trajectories
are necessary and there is not enough GPU memory to process them at once.
"""

import numpy
from beclab import *
from beclab.meters import UncertaintyMeter, getXiSquared


class AveragesCollector:

	def __init__(self, env, constants, grid):
		self._unc = UncertaintyMeter(env, constants, grid)
		self.times = []
		self.n1 = []
		self.n2 = []
		self.i = []

	def prepare(self, **kwds):
		self._unc.prepare(components=kwds['components'],
			ensembles=kwds['ensembles'], psi_type=kwds['psi_type'])

	def __call__(self, t, dt, psi):
		self.times.append(t)

		i, n = self._unc.getEnsembleSums(psi)

		self.i.append(i)
		self.n1.append(n[0])
		self.n2.append(n[1])

	def getData(self):
		return numpy.array(self.times), self.i, self.n1, self.n2


def testUncertainties(a12, gamma12, losses):

	t = 0.2
	callback_dt = 0.002
	N = 55000
	ensembles = 1024

	parameters = dict(
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		a12=a12, a22=95.57,
		gamma12=gamma12, gamma22=7.7e-20)

	if not losses:
		parameters.update(dict(gamma111=0, gamma12=0, gamma22=0))

	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(), **parameters)
	grid = UniformGrid.forN(env, constants, N, (64, 8, 8))

	gs = SplitStepGroundState(env, constants, grid, dt=1e-5)
	evolution = SplitStepEvolution(env, constants, grid, dt=1e-5)
	pulse = Pulse(env, constants, grid, f_rabi=350, f_detuning=-37)

	avc = AveragesCollector(env, constants, grid)

	psi = gs.create((N, 0))
	psi.toWigner(ensembles)

	pulse.apply(psi, math.pi / 2)

	evolution.run(psi, t, callbacks=[avc], callback_dt=callback_dt)
	env.release()

	times, i, n1, n2 = avc.getData()

	return times, i, n1, n2

def combinedTest(N, a12, gamma12, losses):
	t = None
	ii = None
	nn1 = None
	nn2 = None

	for i in xrange(N):
		times, i, n1, n2 = testUncertainties(a12, gamma12, losses)

		if ii is None:
			ii = i
			nn1 = n1
			nn2 = n2
			t = times
		else:
			for j in xrange(len(ii)):
				ii[j] = numpy.concatenate([ii[j], i[j]])
				nn1[j] = numpy.concatenate([nn1[j], n1[j]])
				nn2[j] = numpy.concatenate([nn2[j], n2[j]])

	xi = []
	for j in xrange(len(ii)):
		xi.append(getXiSquared(ii[j], nn1[j], nn2[j]))

	return XYData("a12 = " + str(a12) + ("" if losses else ", no losses"),
		times * 1e3, 10 * numpy.log10(numpy.array(xi)),
		xname="T (ms)", yname="$\\xi^2$ (dB)")


if __name__ == '__main__':

	iterations = 2
	prefix = 'squeezing_feschbach'

	xi1nl = combinedTest(iterations, 80.0, 38.8e-19, False)
	xi2nl = combinedTest(iterations, 85.0, 19.5e-19, False)
	xi3nl = combinedTest(iterations, 90.0, 7.13e-19, False)
	xi4nl = combinedTest(iterations, 95.0, 0.854e-19, False)

	xi1nl.color = 'red'
	xi2nl.color = 'blue'
	xi3nl.color = 'green'
	xi4nl.color = 'black'

	xi1nl.save(prefix + '_80.0_nolosses.json')
	xi2nl.save(prefix + '_85.0_nolosses.json')
	xi3nl.save(prefix + '_90.0_nolosses.json')
	xi4nl.save(prefix + '_95.0_nolosses.json')

	for x in (xi1nl, xi2nl, xi3nl, xi4nl):
		x.linestyle = '--'

	xi1 = combinedTest(iterations, 80.0, 38.8e-19, True)
	xi2 = combinedTest(iterations, 85.0, 19.5e-19, True)
	xi3 = combinedTest(iterations, 90.0, 7.13e-19, True)
	xi4 = combinedTest(iterations, 95.0, 0.854e-19, True)

	xi1.color = 'red'
	xi2.color = 'blue'
	xi3.color = 'green'
	xi4.color = 'black'

	xi1.save(prefix + '_80.0.json')
	xi2.save(prefix + '_85.0.json')
	xi3.save(prefix + '_90.0.json')
	xi4.save(prefix + '_95.0.json')

	XYPlot([xi1nl, xi2nl, xi3nl, xi4nl, xi1, xi2, xi3, xi4],
		location="upper left").save(prefix + '.pdf')
