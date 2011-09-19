"""
Comparison of simulated phase noise with an analytic formula
"""

import numpy
from beclab import *

N = 55000
ensembles = 64

env = envs.cuda()

# breathe-together rehime, no losses
constants = Constants(double=env.supportsDouble(),
	a11=100, a12=50, a22=100,
	gamma111=0, gamma12=0, gamma22=0)
grid = UniformGrid.forN(env, constants, N, (64, 8, 8))

gs = SplitStepGroundState(env, constants, grid, dt=1e-5)
evolution = SplitStepEvolution(env, constants, grid, dt=1e-5)
pulse = Pulse(env, constants, grid, f_rabi=350)

phn = PhaseNoiseCollector(env, constants, grid)
an = AnalyticNoiseCollector(env, constants, grid)

# run simulation
psi = gs.create((N, 0))
psi.toWigner(ensembles)

pulse.apply(psi, math.pi / 2)

evolution.run(psi, 0.5, callbacks=[phn, an], callback_dt=0.01)
env.release()

# save data
times, phnoise = phn.getData()
phnoise = XYData("simulation", times * 1000, phnoise,
	ymin=0, xname="T (ms)", yname="Phase noise, rad")

times, anoise = an.getData()
anoise = XYData("analytic formula", times * 1000, anoise,
	ymin=0, xname="T (ms)", yname="Phase noise, rad")

XYPlot([phnoise, anoise]).save('analytic_noise.pdf')
