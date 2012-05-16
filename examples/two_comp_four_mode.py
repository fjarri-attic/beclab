"""
Create 2-component ground state and plot various plots of its density
"""

import numpy
from beclab import *
from beclab.meters import ProjectionMeter
from beclab.ground_state import GS_INIT_UNIFORM, GS_INIT_RANDOM, \
	GS_ENFORCE_SYM, GS_ENFORCE_ASYM

kb = 1.3806503e-23

def two_well_experimental(constants, grid, dist):

	w = 90 * 2 * numpy.pi
	V_barrier = constants.m * dist ** 2 * w ** 2 / 8

	z = grid.z_full
	potentials = constants.m * z ** 2 * w ** 2 / 2 + \
		V_barrier / 2 * numpy.cos(2 * numpy.pi * z / dist)

	return potentials.astype(constants.scalar.dtype).reshape((1,) + potentials.shape) / constants.hbar

def two_well_gaussian(constants, grid, dist, Vb):

	w = 90 * 2 * numpy.pi

	z = grid.z_full
	potentials = constants.m * z ** 2 * w ** 2 / 2 + \
		Vb * numpy.exp(-(z / dist) ** 2)

	return potentials.astype(constants.scalar.dtype).reshape((1,) + potentials.shape) / constants.hbar

def two_well_4th_order(constants, grid, dist, Vb):

	#Vb = (constants.m * dist ** 2 * constants.wz ** 2) / 8

	z = grid.z_full
	potentials = Vb * (16 * (z / dist) ** 4 - 8 * (z / dist) ** 2 + 1)

	return potentials.astype(constants.scalar.dtype).reshape((1,) + potentials.shape) / constants.hbar

def single_well(constants, grid, dist):
	z = grid.z_full
	w = 90 * 2 * numpy.pi
	potentials = constants.m * z ** 2 * w ** 2 / 2
	return potentials.astype(constants.scalar.dtype).reshape((1,) + potentials.shape) / constants.hbar

def getMu(psi, **c):
	N = c['N']
	Ez = c['Ez']
	return psi.interaction_meter.getMuTotal() / N / Ez

def getE(psi, **c):
	N = c['N']
	Ez = c['Ez']
	return psi.interaction_meter.getETotal() / N / Ez

def getOverlap4(env, grid, psiA, psiB, c1=0, c2=0, **c):
	lz = c['lz']

	dz = grid.dz
	dA = env.fromDevice(psiA.data)
	dB = env.fromDevice(psiB.data)

	dA1 = numpy.abs(dA[c1, 0]) ** 2
	dA2 = numpy.abs(dA[c2, 0]) ** 2
	dB2 = numpy.abs(dB[c2, 0]) ** 2

	gammaAA = (dA1 * dA2 * dz).sum() #* lz
	gammaAB = (dA1 * dB2 * dz).sum() #* lz

	return gammaAA, gammaAB

def getOverlap6(env, grid, psi, comp=0, **c):
	dz = grid.dz
	d = env.fromDevice(psi.data)[comp, 0]
	return (numpy.abs(d) ** 6 * dz).sum()

def getInteraction(env, constants, grid, psi1, psi2, comp=0, **c):
	dz = grid.dz
	d1 = env.fromDevice(psi1.data)[comp,0]
	d2 = env.fromDevice(psi2.data)[comp,0]
	e = grid.energy
	p = grid.potentials[0]

	res = numpy.sum(d1.conj() * (numpy.fft.ifft(numpy.fft.fft(d2) * e) + p * d2) * dz) * constants.hbar
	return res.real

def plotDensity(psi1, psi2, n1, n2, name):
	z = psi1._grid.z * 1e6
	prj = ProjectionMeter.forPsi(psi1)
	z_proj1 = prj.getZ(psi1) / 1e6 # cast to micrometers^-1
	z_proj2 = prj.getZ(psi2) / 1e6 # cast to micrometers^-1
	datas = [z_proj1]

	plots = []
	for comp in xrange(psi1.components):
		plots += [
			XYData(n1 + ", component " + str(comp + 1), z, z_proj1[comp],
				xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0),
			XYData(n2 + ", component " + str(comp + 1), z, z_proj2[comp],
				xname="Z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)", ymin=0),
		]

	XYPlot(plots).save(name)

def getOverlaps(components=1, rk5=True, harmonic=False, random=False):
	env = envs.cuda(device_num=1)

	box_size = 2e-5
	a12 = 80.7

	constants = Constants(double=env.supportsDouble(), fx=42e3, fy=42e3, fz=1650,
		a12=a12, use_effective_area=True)

#	mu = constants.muTF(4000, dim=1)
#	I2 = 16.0 / 15 * (mu ** 2.5) / (numpy.sqrt(constants.m * constants.wz ** 2 / 2) * ((constants.g[0,0] * 4000) **  2))
#	print I2
#	print 500e-9 * 1.38e-23 / (constants.g[0,0] * 200 * 1e5)
#	env.release()
#	exit(0)

	lz = numpy.sqrt(constants.hbar / (constants.m * constants.wz))
	Ez = constants.hbar * constants.wz

	# dimensionless parameters
	sigma = 4.
	Vb = 50.
	#gN = 10.
	#N = gN * Ez * lz / constants.g[0, 0]
	N = 4000.

	Vb_dim = Vb * Ez
	#dist = sigma * lz
	dist = 0.5e-5

	cd = dict(lz=lz, Ez=Ez, N=N)

	grid = UniformGrid(env, constants, (8192,), (box_size,),
		potentials_func=lambda c, g: two_well_4th_order(c, g, dist, Vb_dim))

	if rk5:
		gs = RK5IPGroundState(env, constants, grid, Nscale=N, eps=1e-9, atol_coeff=1e-4)
	else:
		gs = SplitStepGroundState(env, constants, grid, dt=1e-6)

	if rk5:
		params = dict(relative_precision=1e-2)
	else:
		params = dict(precision=1e-8)
	params.update(dict(gs_init=GS_INIT_UNIFORM if random else GS_INIT_UNIFORM))

	population = (N,) if components == 1 else (N/2, N/2)
	psi_s = gs.create(population, enforce=GS_ENFORCE_SYM, **params)
	psi_a = gs.create(population, enforce=GS_ENFORCE_ASYM, **params)

	print "Sym, mu =", getMu(psi_s, **cd), "E =", getE(psi_s, **cd)
	print "Asym, mu =", getMu(psi_a, **cd), "E =", getE(psi_a, **cd)

	data_s = env.fromDevice(psi_s.data) / numpy.sqrt(N / components)
	data_a = env.fromDevice(psi_a.data) / numpy.sqrt(N / components)
	psi_s.fillWith(data_s)
	psi_a.fillWith(data_a)

	for comp in xrange(components):
		K_ss = getInteraction(env, constants, grid, psi_s, psi_s, comp=comp, **cd)
		K_aa = getInteraction(env, constants, grid, psi_a, psi_a, comp=comp, **cd)
		print "component", comp + 1, ": sKs =", K_ss / Ez, "Ez =", K_ss / kb * 1e9, "nK"
		print "component", comp + 1, ": aKa =", K_aa / Ez, "Ez =", K_aa / kb * 1e9, "nK"

	"""
	gamma_mm, gamma_pp, gamma_pm = getOverlap4(env, grid, psi_a, psi_s, c1=0, c2=0, **cd)
	gamma_mm *= constants.g[0, 0] * (N / components) / Ez
	gamma_pp *= constants.g[0, 0] * (N / components) / Ez
	gamma_pm *= constants.g[0, 0] * (N / components) / Ez
	beta_p = getMu(psi_s, **cd)
	beta_m = getMu(psi_a, **cd)
	dgamma = gamma_mm - gamma_pp
	dbeta = beta_m - beta_p
	A = (10 * gamma_pm - gamma_mm - gamma_pp) / 4
	B = dbeta - dgamma / 2
	C = (gamma_mm + gamma_pp - 2 * gamma_pm) / 4
	F = (beta_m + beta_p) / 2 - gamma_pm
	print F, A, dgamma, dbeta, C
	"""

	plotDensity(psi_s, psi_a, "symmetric", "antisymmetric", "psis_as.pdf")

	psiA = WavefunctionSet(env, constants, grid, components=components, ensembles=1)
	psiB = WavefunctionSet(env, constants, grid, components=components, ensembles=1)
	dataA = (env.fromDevice(psi_s.data) + env.fromDevice(psi_a.data)) / numpy.sqrt(2.0)
	dataB = (env.fromDevice(psi_s.data) - env.fromDevice(psi_a.data)) / numpy.sqrt(2.0)
	psiA.fillWith(dataA)
	psiB.fillWith(dataB)

	for c1 in xrange(components):
		for c2 in xrange(c1, components):
			I_AA, I_AB = getOverlap4(env, grid, psiA, psiB, c1=c1, c2=c2, **cd)
			print "components", c1 + 1, c2 + 1, ": I_AA =", I_AA * lz, "I_AB =", I_AB * lz

	plotDensity(psiA, psiB, "well A", "well B", "psis_12.pdf")

	for comp in xrange(components):
		K_AA = getInteraction(env, constants, grid, psiA, psiA, comp=comp, **cd)
		K_AB = getInteraction(env, constants, grid, psiA, psiB, comp=comp, **cd)
		print "component", comp + 1, ": A_K_A =", K_AA / Ez, "Ez =", K_AA / kb * 1e9, "nK"
		print "component", comp + 1, ": A_K_B =", K_AB / Ez, "Ez =", K_AB / kb * 1e9, "nK"

	# comparison of relative effect
	gamma12_t = 1.77e-12 * 1e-6 # theorist's, m^3/s
	gamma22_t = 8.1e-14 * 1e-6 / 4 # theorists's, m^3/s
	gamma111_t = 5.4e-30 * 1e-12 / 6 # theorist's, m^3/s

	kappa1 = getInteraction(env, constants, grid, psiA, psiB, comp=0, **cd) / constants.hbar
	I_AA, _ = getOverlap4(env, grid, psiA, psiB, c1=0, c2=0, **cd)
	I_AA_12, _ = getOverlap4(env, grid, psiA, psiB, c1=0, c2=1, **cd)
	I_AA_22, _ = getOverlap4(env, grid, psiA, psiB, c1=1, c2=1, **cd)
	I_AAA = getOverlap6(env, grid, psiA, comp=0, **cd)
	g11_N = constants.g[0, 0] * (N / components) * I_AA / constants.hbar
	gamma12 = gamma12_t * (N / components) * I_AA / constants.getEffectiveArea()
	gamma22 = gamma22_t * (N / components) * I_AA / constants.getEffectiveArea()
	gamma111 = gamma111_t * (N / components) ** 2 * I_AAA / (constants.getEffectiveArea() ** 2)
	#print kappa1, g11_N, gamma12, gamma22, gamma111
	print I_AA, I_AA_12, I_AA_22, I_AAA
	print kappa1 / g11_N, gamma12 / g11_N, gamma22 / g11_N, gamma111 / g11_N

	env.release()

	#return g_mm, g_pp, g_pm

def testAnanikian():
	Vbs = numpy.linspace(1, 15, 15)
	g_mm = []
	g_pp = []
	g_pm = []
	for Vb in Vbs:
		mm, pp, pm = getOverlaps(Vb)
		g_mm.append(mm)
		g_pp.append(pp)
		g_pm.append(pm)

	XYPlot([
		XYData("g--", Vbs, numpy.array(g_mm), xname="Vb", yname="g"),
		XYData("g++", Vbs, numpy.array(g_pp), xname="Vb", yname="g"),
		XYData("g+-", Vbs, numpy.array(g_pm), xname="Vb", yname="g"),
	]).save('g_mm_pp_pm.pdf')

if __name__ == '__main__':
	getOverlaps(components=2)
