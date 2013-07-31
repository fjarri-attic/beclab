"""
Ramsey sequence and cloud snapshots during the evolution
"""

import numpy
from beclab import *

N = 150000

env = envs.cuda()
constants = Constants(double=env.supportsDouble())
grid = UniformGrid.forN(env, constants, N, (128, 16, 16))

gs = SplitStepGroundState(env, constants, grid, dt=1e-5)
evolution = RK5IPEvolution(env, constants, grid, eps=1e-4, atol_coeff=1e-1)
pulse = Pulse(env, constants, grid, f_rabi=350, f_detuning=41)
a = SurfaceProjectionCollector(env, constants, grid, pulse=pulse)

# run simulation
psi = gs.create((N, 0))
pulse.apply(psi, theta=numpy.pi / 2)
evolution.run(psi, 0.119, callbacks=[a], callback_dt=0.005)
env.release()

# build plot
times, xy, yz, xz = a.getData()
times = [str(int(x * 1000)) for x in times]

# right after pulse second component has all the population,
# so we are estimating maximum density based on its center
max_density = yz[1, 0, grid.shape[0] / 2, grid.shape[1] / 2]
for comp in (0, 1):
	hms = []
	for t, hm in zip(times[:-1], yz[comp,:-1]):
		hms.append(HeightmapData(t, hm.transpose(),
			xmin=grid.z[0], xmax=grid.z[-1],
			ymin=grid.y[0], ymax=grid.y[-1],
			zmin=0, zmax=max_density))

	EvolutionPlot(hms, shape=(6, 4)).save('evolution_snapshots_comp' + str(comp + 1) + '.pdf')
