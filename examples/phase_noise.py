"""
Comparison of GPE, GPE with classical noise, Wigner representation
and classical noise + Wigner
"""

import numpy
from beclab import *

N = 55000
ensembles = 256

def test(classical_noise, wigner):

	if wigner and classical_noise:
		name = "Wigner + classical noise"
	elif wigner:
		name = "Wigner"
	elif classical_noise:
		name = "Classical noise"
	else:
		name = "GPE"

	print "\n* Running:", name, "\n"

	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(),
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		a12=97.99, a22=95.57,
		gamma12=1.53e-20, gamma22=7.7e-20)
	grid = UniformGrid.forN(env, constants, N, (64, 8, 8))

	gs = SplitStepGroundState(env, constants, grid, dt=1e-5)
	evolution = SplitStepEvolution(env, constants, grid, dt=1e-5)
	pulse = Pulse(env, constants, grid, f_rabi=350, f_detuning=-37)

	v = VisibilityCollector(env, constants, grid, verbose=True)
	pn = ParticleNumberCollector(env, constants, grid, verbose=True, pulse=pulse)
	phn = PhaseNoiseCollector(env, constants, grid, verbose=True)
	pzn = PzNoiseCollector(env, constants, grid, verbose=True)

	# run simulation
	psi = gs.create((N, 0))
	if wigner:
		psi.toWigner(ensembles)
	else:
		psi.createEnsembles(ensembles)

	if classical_noise:
		pulse.apply(psi, math.pi / 2, theta_noise=0.02)
	else:
		pulse.apply(psi, math.pi / 2)

	evolution.run(psi, 1.0, callbacks=[v, pn, phn, pzn], callback_dt=0.01)
	env.release()

	# save data
	times, vis = v.getData()
	vis = XYData(name, times * 1000, vis, ymin=0, ymax=1,
		xname="T (ms)", yname="$\\mathcal{V}$")

	times, phnoise = phn.getData()
	phnoise = XYData(name, times * 1000, phnoise, ymin=0, xname="T (ms)", yname="Phase noise, rad")

	times, pznoise = pzn.getData()
	pznoise = XYData(name, times * 1000, pznoise, ymin=0, xname="T (ms)", yname="$P_z$ noise, rad")

	return vis, phnoise, pznoise

if __name__ == '__main__':

	results = [
		test(False, False),
		test(False, True),
		test(True, False),
		test(True, True)
	]

	XYPlot([res[0] for res in results]).save('phase_noise_visibility.pdf')
	XYPlot([res[1] for res in results]).save('phase_noise_phnoise.pdf')
	XYPlot([res[2] for res in results]).save('phase_noise_pznoise.pdf')
