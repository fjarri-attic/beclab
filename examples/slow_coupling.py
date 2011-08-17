"""
Slow coupling using GPE simulated oscillator
"""

import numpy
from beclab import *

N = 50000

env = envs.cpu()
constants = Constants(double=env.supportsDouble())
grid = UniformGrid.forN(constants, N, (64, 8, 8))

gs = SplitStepGroundState(env, constants, grid, dt=1e-5)
evolution = SplitStepEvolution(env, constants, grid, dt=1e-5, f_rabi=10, f_detuning=0)

v = VisibilityCollector(env, constants, grid, verbose=True)
p = ParticleNumberCollector(env, constants, grid, pulse=None, verbose=True)

# Run simulation
psi = gs.create((N, 0))
evolution.run(psi, 0.1, callbacks=[v, p], callback_dt=0.001)
env.release()

# Create plots
times, vis = v.getData()
XYPlot([
	XYData("Visibility", times, vis, ymin=0, ymax=1, xname="T (s)", yname="$\\mathcal{V}$")
]).save('slow_coupling_visibility.pdf')

times, N, Ntotal = p.getData()
XYPlot([
	XYData("Population ratio", times, (N[0] - N[1]) / Ntotal, ymin=-1, ymax=1,
		xname="T (s)", yname="Population ratio")
]).save('slow_coupling_population.pdf')



