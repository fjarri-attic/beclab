import numpy
import time
from beclab import *
from beclab.meters import ParticleStatistics


def testAxial(gpu):
	env = envs.cuda() if gpu else envs.cpu()
	try:
		return runTest(env)
	finally:
		env.release()

def runTest(env):

	N = 150000
	constants = Constants(double=env.supportsDouble(), gamma12=0, gamma22=0, gamma111=0)
	grid = UniformGrid.forN(env, constants, N, (64, 8, 8))

	gs = SplitStepGroundState(env, constants, grid)
	evolution = SplitStepEvolution(env, constants, grid, dt=4e-5)
	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350)
	a = AxialProjectionCollector(env, constants, grid, matrix_pulse=True, pulse=pulse)
	p = ParticleNumberCollector(env, constants, grid, matrix_pulse=True, pulse=pulse, verbose=True)

	s = ParticleStatistics(env, constants, grid, components=2)

	# experiment
	psi = gs.create((N, 0))

	pulse.apply(psi, theta=0.5 * numpy.pi, matrix=True)

#	print s.getN(psi)

	t1 = time.time()
	evolution.run(psi, time=0.1, callbacks=[a, p], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"
#	print s.getN(psi)

	times, heightmap = a.getData()

	res = HeightmapPlot(
		HeightmapData("test", heightmap,
			xmin=0, xmax=100,
			ymin=grid.z[0] * 1e6,
			ymax=grid.z[-1] * 1e6,
			zmin=-1, zmax=1,
			xname="T (ms)", yname="z ($\\mu$m)", zname="Spin projection")
	)

	return res

for gpu in (False, True,):
	testAxial(gpu).save("axial_view_" + ('GPU' if gpu else 'CPU') + ".pdf")
