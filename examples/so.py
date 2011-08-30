import numpy
from beclab import *
from beclab.constants import SOConstants
from beclab.ground_state import SOGroundState
from beclab.meters import ParticleStatistics

so_constants = SOConstants(g_ratio=1.5, g_strength=0.2, lambda_SO=20)

env = envs.cuda()
constants = Constants(double=env.supportsDouble(),
	g_intra=so_constants.g_intra, g_inter=so_constants.g_inter,
	lambda_R=so_constants.lambda_R, m=so_constants.m,
	fx=so_constants.f_perp, fy=so_constants.f_perp)

box_size = (so_constants.a_perp * 2, so_constants.a_perp * 2)
grid = UniformGrid(constants, (64, 64), box_size)

gs = SOGroundState(env, constants, grid, dt=1e-6, components=2, precision=1e-8)
stats = ParticleStatistics(env, constants, grid, components=2, ensembles=1)

N = so_constants.N
psi = gs.create((N / 2, N / 2))

data = env.fromDevice(stats.getDensity(psi))
env.release()

data *= so_constants.a_perp ** 2
plot_params = dict(
	xmin=-box_size[1] / 2 / so_constants.a_perp,
	xmax=box_size[1] / 2 / so_constants.a_perp,
	ymin=-box_size[0] / 2 / so_constants.a_perp,
	ymax=box_size[0] / 2 / so_constants.a_perp,
	xname="$x / a_{\\perp}$", yname="$y / a_{\\perp}$", zname="density",
	zmin=0, zmax=data.max()
)

HeightmapPlot(HeightmapData("|up>", data[0, 0], **plot_params)).save('so_a.pdf')
HeightmapPlot(HeightmapData("|down>", data[1, 0], **plot_params)).save('so_b.pdf')
