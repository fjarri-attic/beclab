import numpy
from beclab import *
from beclab.ground_state import SOGroundState
from beclab.meters import ParticleStatistics

# dimensionless parameters
g_ratio = 0.5 # g_interspecies / g_intraspecies
g_strength = 0.2 # g_intraspecies * (N - 1) / (hbar * w_perp) / a_perp ** 2
lambda_SO = 20 # dimensionless coupling strength, a_perp / a_lambda

# real parameters
r_bohr = 5.2917720859e-11
hbar = 1.054571628e-34
m = 1.443160648e-25 # Rb-87

a = 100 * r_bohr # scattering length for intra-species interaction (guess)
f_z = 1.9e3 # frequency of 2D-creating confinement (guess)
f_perp = 20 # frequency of the actual trap (guess)

w_perp = f_perp * numpy.pi * 2
a_perp = numpy.sqrt(hbar / (m * w_perp))
a_z = numpy.sqrt(hbar / (m * f_z * numpy.pi * 2))
g_intra = numpy.sqrt(numpy.pi * 8) * (hbar ** 2 / m) * (a / a_z)
g_inter = g_intra * g_ratio
N = g_strength * hbar * w_perp * a_perp ** 2 / g_intra + 1
a_lambda = a_perp / lambda_SO
lambda_R = hbar ** 2 / (m * a_lambda)

env = envs.cpu()
constants = Constants(double=env.supportsDouble(),
	g_intra=g_intra, g_inter=g_inter, lambda_R=lambda_R, m=m,
	fx=f_perp, fy=f_perp)

box_size = (a_perp * 2, a_perp * 2)

grid = UniformGrid(constants, (64, 64), box_size)
gs = SOGroundState(env, constants, grid, dt=1e-6)
stats = ParticleStatistics(env, constants, grid, components=2, ensembles=1)

print "N =", N
psi = gs.create((N / 2, N / 2))

data = env.fromDevice(stats.getDensity(psi))
env.release()

data *= a_perp ** 2
HeightmapPlot(HeightmapData("|up>", data[0, 0],
	xmin=-box_size[1] / 2 / a_perp, xmax=box_size[1] / 2 / a_perp,
	ymin=-box_size[0] / 2 / a_perp, ymax=box_size[0] / 2 / a_perp,
	xname="$x / a_{\\perp}$", yname="$y / a_{\\perp}$", zname="density",
	zmin=0, zmax=data.max())).save('so_a.pdf')
HeightmapPlot(HeightmapData("|down>", data[1, 0],
	xmin=-box_size[1] / 2 / a_perp, xmax=box_size[1] / 2 / a_perp,
	ymin=-box_size[0] / 2 / a_perp, ymax=box_size[0] / 2 / a_perp,
	xname="$x / a_{\\perp}$", yname="$y / a_{\\perp}$", zname="density",
	zmin=0, zmax=data.max())).save('so_b.pdf')
