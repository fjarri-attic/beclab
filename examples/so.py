import numpy
from beclab import *
from beclab.constants import SOConstants
from beclab.ground_state_so import SOGroundState
from beclab.meters import ParticleStatistics
import time

so_constants = SOConstants(g_ratio=1.5, g_strength=0.2, lambda_SO=20)

env = envs.cuda()
constants = Constants(double=env.supportsDouble(),
	g_intra=so_constants.g_intra, g_inter=so_constants.g_inter,
	lambda_R=so_constants.lambda_R, m=so_constants.m,
	fx=so_constants.f_perp, fy=so_constants.f_perp)

box_size = (so_constants.a_perp * 4, so_constants.a_perp * 4)
grid = UniformGrid(constants, (512, 512), box_size)

t_scale = 1.0 / (so_constants.f_perp * numpy.pi * 2)
print "time scale = ", t_scale
gs = SOGroundState(env, constants, grid, dt=t_scale / 50, components=2, precision=1e-10)
stats = ParticleStatistics(env, constants, grid, components=2, ensembles=1)

N = so_constants.N
print "target N = ", N
t1 = time.time()
psi = gs.create((N / 2, N / 2))
print "creation time = ", time.time() - t1

Ns = stats.getN(psi)
print "final N = ", Ns
E = stats.getSOEnergy(psi) / (constants.hbar * so_constants.f_perp * numpy.pi * 2)
print "final energy = ", E.sum() * N, "(", E.sum(), " * N)"
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

suffix = "{g_ratio}_{g_strength}_{lambda_SO}".format(
	g_ratio=so_constants.g_ratio, g_strength=so_constants.g_strength,
	lambda_SO=so_constants.lambda_SO)
HeightmapPlot(HeightmapData("|up>", data[0, 0], **plot_params), aspect=1).save(
	'so_a_' + suffix + '.pdf')
HeightmapPlot(HeightmapData("|down>", data[1, 0], **plot_params), aspect=1).save(
	'so_b_' + suffix + '.pdf')
