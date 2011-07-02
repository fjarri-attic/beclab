import numpy

from .helpers import *
from .constants import *


class ParticleStatistics(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		#self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
		self._reduce = createReduce(env, constants.scalar.dtype)
		self._creduce = createReduce(env, constants.complex.dtype)

		self._potentials = getPotentials(env, constants, grid)

		if isinstance(grid, HarmonicGrid):
			self._energy = getHarmonicEnergy(env, constants, grid)
		elif isinstance(grid, UniformGrid):
			self._energy = getPlaneWaveEnergy(env, constants, grid)

		self._dV = grid.get_dV(env)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernel_template = """
			EXPORTED_FUNC void interaction(GLOBAL_MEM COMPLEX *interaction,
				GLOBAL_MEM COMPLEX *a_state, GLOBAL_MEM COMPLEX *b_state)
			{
				DEFINE_INDEXES;
				interaction[index] = complex_mul(a_state[index], conj(b_state[index]));
			}

			EXPORTED_FUNC void density(GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *state, int ensembles, SCALAR modifier)
			{
				DEFINE_INDEXES;
				res[index] = (squared_abs(state[index]) - modifier) / ensembles;
			}

			EXPORTED_FUNC void invariant(GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *xstate, GLOBAL_MEM COMPLEX *kstate,
				GLOBAL_MEM SCALAR *potentials,
				SCALAR g_by_hbar, int coeff)
			{
				DEFINE_INDEXES;

				SCALAR potential = potentials[cell_index];

				SCALAR n = squared_abs(xstate[index]);
				COMPLEX differential = complex_mul(conj(xstate[index]), kstate[index]);
				SCALAR nonlinear = n * (potential + g_by_hbar * n / coeff);

				// differential.y will be equal to 0, because \psi * D \psi is a real number
				res[index] = nonlinear + differential.x;
			}

			EXPORTED_FUNC void invariant2comp(GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *xstate1, GLOBAL_MEM COMPLEX *kstate1,
				GLOBAL_MEM COMPLEX *xstate2, GLOBAL_MEM COMPLEX *kstate2,
				GLOBAL_MEM SCALAR *potentials,
				SCALAR g11_by_hbar, SCALAR g22_by_hbar,
				SCALAR g12_by_hbar, int coeff)
			{
				DEFINE_INDEXES;

				SCALAR potential = potentials[cell_index];

				SCALAR n1 = squared_abs(xstate1[index]);
				SCALAR n2 = squared_abs(xstate2[index]);

				COMPLEX differential1 =
					complex_mul(conj(xstate1[index]), kstate1[index]);
				COMPLEX differential2 =
					complex_mul(conj(xstate2[index]), kstate2[index]);

				SCALAR nonlinear1 = n1 * (potential +
					g11_by_hbar * n1 / coeff +
					g12_by_hbar * n2 / coeff);
				SCALAR nonlinear2 = n2 * (potential +
					g12_by_hbar * n1 / coeff +
					g22_by_hbar * n2 / coeff);

				// differential.y will be equal to 0, because \psi * D \psi is a real number
				res[index] = nonlinear1 + differential1.x +
					nonlinear2 + differential2.x;
			}

			EXPORTED_FUNC void multiplyScalars(GLOBAL_MEM SCALAR *data, GLOBAL_MEM SCALAR *coeffs,
				int ensembles)
			{
				DEFINE_INDEXES;
				SCALAR coeff_val = coeffs[index];
				SCALAR data_val;

				for(int i = 0; i < ensembles; i++)
				{
					data_val = data[index + i * ${g.size}];
					data[index + i * ${g.size}] = data_val * coeff_val;
				}
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants, self._grid)

		self._kernel_interaction = self._program.interaction
		self._kernel_invariant = self._program.invariant
		self._kernel_invariant2comp = self._program.invariant2comp
		self._kernel_density = self._program.density
		self._kernel_multiplyScalars = self._program.multiplyScalars

	def _cpu__kernel_calculateInteraction(self, _, res, data0, data1):
		self._env.copyBuffer(data0 * data1.conj(), dest=res)

	def _cpu__kernel_density(self, _, density, data, coeff, modifier):
		self._env.copyBuffer((numpy.abs(data) ** 2 - modifier) / coeff, dest=density)

	def _cpu__kernel_multiplyScalars(self, _, res, coeffs, ensembles):
		res.flat *= numpy.tile(coeffs.flat, ensembles)

	def getVisibility(self, psi0, psi1):
		N0 = self.getN(psi0)
		N1 = self.getN(psi1)
		interaction = self._getInteraction(psi0, psi1)
		self._kernel_multiply(interaction.size, interaction, self._dV)

		ensembles = psi0.shape[0]
		interaction = numpy.abs(self._creduce(interaction)) / ensembles

		return 2.0 * interaction / (N0 + N1)

	def getDensity(self, psi, coeff=1):
		if psi.type == WIGNER:
			raise NotImplementedError()
			# Need to find modifier value for harmonic case
			#modifier = self._grid.modes / (2.0 * self._constants.V)
		else:
			modifier = 0.0

		density = self._env.allocate(psi.shape, self._constants.scalar.dtype)
		self._kernel_density(psi.size, density, psi.data, numpy.int32(coeff),
			self._constants.scalar.cast(modifier))
		return density

	def getAverageDensity(self, psi):
		# Using psi.size and .shape instead of grid here, to make it work
		# for both x- and mode-space.
		ensembles = psi.shape[0]
		density = self.getDensity(psi, coeff=ensembles)
		average_density = self._reduce.sparse(density, final_length=psi.size / ensembles,
			final_shape=psi.shape[1:])
		return average_density

	def getAveragePopulation(self, psi):
		density = self.getAverageDensity(psi)
		if not psi.in_mspace:
			self._kernel_multiplyScalars(density.size, density, self._dV, numpy.int32(psi.shape[0]))
		return density

	def _getInvariant(self, psi, coeff, N):

		# TODO: work out the correct formula for Wigner function's E/mu
		if psi.type != CLASSICAL:
			raise NotImplementedError()

		# If N is not known beforehand, we have to calculate it first
		if N is None:
			N = self.getN(psi)

		psi.toMSpace()
		mdata = self._env.copyBuffer(psi0.data)
		psi.toXSpace()

		cast = self._constants.scalar.cast
		g = cast(self._constants.g[psi.comp, psi.comp])

		res = self._env.allocate(psi.shape, dtype=self._constants.complex.dtype)
		self._kernel_invariant(psi.size, res,
			psi.data, mdata,
			self._potentials, self._energy,
			g, numpy.int32(coeff))
		self._kernel_multiply(res.size, res, self._dV)
		return self._reduce(res) / psi.shape[0] / N * self._constants.hbar

	def _getInvariant2comp(self, psi0, psi1, coeff, N):

		# TODO: work out the correct formula for Wigner function's E/mu
		if psi0.type != CLASSICAL or psi1.type != CLASSICAL:
			raise NotImplementedError()

		# If N is not known beforehand, we have to calculate it first
		if N is None:
			N = self.getN(psi0) + self.getN(psi1)

		psi0.toMSpace()
		psi1.toMSpace()
		mdata0 = self._env.copyBuffer(psi0.data)
		mdata1 = self._env.copyBuffer(psi1.data)
		psi0.toXSpace()
		psi1.toXSpace()

		g = self._constants.g
		cast = self._constants.scalar.cast
		g00 = cast(g[psi0.comp, psi0.comp])
		g01 = cast(g[psi0.comp, psi1.comp])
		g11 = cast(g[psi1.comp, psi1.comp])

		res = self._env.allocate(psi0.shape, dtype=self._constants.complex.dtype)
		self._kernel_invariant2comp(psi0.size, res,
			psi0.data, psi1.data,
			mdata0, mdata1,
			self._potentials, self._energy,
			g00, g01, g11, numpy.int32(coeff))
		self._kernel_multiply(res.size, res, self._dV)
		return self._reduce(res) / psi0.shape[0] / N * self._constants.hbar

	def _getInteraction(self, psi0, psi1):
		interaction = self._env.allocate(psi0.shape, self._constants.complex.dtype)
		self._kernel_interaction(psi0.size, interaction, psi0.data, psi1.data)
		return interaction

	def getPhaseNoise(self, psi0, psi1):
		"""
		Warning: this function considers spin distribution ellipse to be horizontal,
		which is not always so.
		"""

		ensembles = psi0.shape[0]
		get = self._env.fromDevice
		reduce = self._reduce
		creduce = self._creduce

		i = self._getInteraction(psi0, psi1)

		i = get(creduce(i, ensembles)) # Complex numbers {S_xj + iS_yj, j = 1..N}
		phi = numpy.angle(i) # normalizing

		# Center of the distribution can be shifted to pi or -pi,
		# making mean() return incorrect values.
		# The following approximate method will allow us to shift the center to zero
		# It will work only if the maximum of the distribution is clearly
		# distinguished; otherwise it can give anything as a result

		Pperp = numpy.exp(1j * phi) # transforming Pperp to distribution on the unit circle
		Pmean = Pperp.mean() # Center of masses is supposed to be close to the center of distribution

		# Normalizing the direction to the center of masses
		# Now angle(Pmean) ~ proper mean of Pperp
		Pmean /= numpy.abs(Pmean)

		# Shifting the distribution
		Pcent = Pperp * Pmean.conj()
		phi_centered = numpy.angle(Pcent)

		return phi_centered.std()

	def getPzNoise(self, psi0, psi1):
		ensembles = psi0.shape[0]
		get = self._env.fromDevice
		reduce = self._reduce

		n0 = self.getDensity(psi0)
		n1 = self.getDensity(psi1)

		n0 = get(reduce(n0, ensembles))
		n1 = get(reduce(n1, ensembles))

		Pz = (n0 - n1) / (n0 + n1)

		return Pz.std()

	def getN(self, psi):
		p = self.getAveragePopulation(psi)
		return self._reduce(p)

	def getEnergy(self, psi, N=None):
		return self._getInvariant(psi, 2, N)

	def getMu(self, psi, N=None):
		return self._getInvariant(psi, 1, N)

	def getEnergy2comp(self, psi0, psi1, N=None):
		return self._getInvatiant2comp(psi0, psi1, 2, N)

	def getMu2comp(self, psi0, psi1, N=None):
		return self._getInvariant2comp(psi0, psi1, 1, N)


class DensityProfile:

	def __init__(self, env, constants, grid):
		self._env = env
		self._constants = constants.copy()
		self._grid = grid.copy()
		self._reduce = createReduce(env, constants.scalar.dtype)
		self._stats = ParticleStatistics(env, constants, grid)

	def getXY(self, psi):
		p = self._stats.getAveragePopulation(psi)
		x = self._grid.shape[2]
		y = self._grid.shape[1]
		# FIXME: need to properly account for non-uniform dV (like getZ)
		return NotImplementedError()
		#return self._env.fromDevice(self._reduce.sparse(p, final_length=x * y), shape=(y, x))

	def getYZ(self, psi):
		p = self._stats.getAveragePopulation(psi)
		y = self._grid.shape[1]
		z = self._grid.shape[0]
		# FIXME: need to properly account for non-uniform dV (like getZ)
		return NotImplementedError()
		#return self._env.fromDevice(self._reduce(p, final_length=y * z), shape=(z, y))

	def getZ(self, psi):
		p = self._stats.getAveragePopulation(psi)
		z = self._grid.shape[0]
		return self._env.fromDevice(self._reduce(p, final_length=z)) / self._grid.dz


class Slice:

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants
		self._stats = ParticleStatistics(env, constants)

	def getXY(self, state):
		density = self._stats.getAverageDensity(state)
		temp = self._env.fromDevice(density)
		return temp[self._constants.nvz / 2,:,:]

	def getYZ(self, state):
		density = self._stats.getAverageDensity(state)
		temp = self._env.fromDevice(density).transpose((2, 0, 1))
		return temp[self._constants.nvx / 2,:,:]


class Uncertainty:

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants
		self._stats = ParticleStatistics(env, constants)
		self._reduce = createReduce(env, constants.scalar.dtype)
		self._creduce = createReduce(env, constants.complex.dtype)

	def getNstddev(self, state):
		ensembles = state.size / self._constants.cells
		get = self._env.fromDevice
		reduce = self._reduce
		dV = self._constants.dV

		n = self._stats.getDensity(state)
		n = get(reduce(n, ensembles)) * dV

		return numpy.std(n)

	def getSpins(self, state1, state2):
		ensembles = state1.size / self._constants.cells
		get = self._env.fromDevice
		reduce = self._reduce
		creduce = self._creduce
		dV = self._constants.dV

		i = self._stats._getInteraction(state1, state2)
		n1 = self._stats.getDensity(state1)
		n2 = self._stats.getDensity(state2)

		i = get(creduce(i, ensembles)) * dV
		n1 = get(reduce(n1, ensembles)) * dV
		n2 = get(reduce(n2, ensembles)) * dV

		# Si for each trajectory
		Si = [i.real, i.imag, 0.5 * (n1 - n2)]
		S = numpy.sqrt(Si[0] ** 2 + Si[1] ** 2 + Si[2] ** 2)
		phi = numpy.arctan2(Si[1], Si[0])
		yps = numpy.arccos(Si[2] / S)

		return phi, yps

	def getXiSquared(self, state1, state2):
		"""Get squeezing coefficient; see Yun Li et al, Eur. Phys. J. B 68, 365-381 (2009)"""

		ensembles = state1.size / self._constants.cells
		get = self._env.fromDevice
		reduce = self._reduce
		creduce = self._creduce
		dV = self._constants.dV

		i = self._stats._getInteraction(state1, state2)
		n1 = self._stats.getDensity(state1)
		n2 = self._stats.getDensity(state2)

		i = get(creduce(i, ensembles)) * dV
		n1 = get(reduce(n1, ensembles)) * dV
		n2 = get(reduce(n2, ensembles)) * dV

		return self._getXiSquared(i, n1, n2)

	def _getXiSquared(self, i, n1, n2):

		Si = [i.real, i.imag, 0.5 * (n1 - n2)] # S values for each trajectory
		avgs = [x.mean() for x in Si] # <S_i>, i=x,y,z

		# \Delta_{ii} = 2 \Delta S_i^2
		deltas = numpy.array([[(x * y + y * x).mean() - 2 * x.mean() * y.mean() for x in Si] for y in Si])

		S = numpy.sqrt(avgs[0] ** 2 + avgs[1] ** 2 + avgs[2] ** 2) # <S>
		phi = numpy.arctan2(avgs[1], avgs[0]) # azimuthal angle of S
		yps = numpy.arccos(avgs[2] / S) # polar angle of S

		sin = numpy.sin
		cos = numpy.cos

		A = (sin(phi) ** 2 - cos(yps) ** 2 * cos(phi) ** 2) * 0.5 * deltas[0, 0] + \
			(cos(phi) ** 2 - cos(yps) ** 2 * sin(phi) ** 2) * 0.5 * deltas[1, 1] - \
			sin(yps) ** 2 * 0.5 * deltas[2, 2] - \
			0.5 * (1 + cos(yps) ** 2) * sin(2 * phi) * deltas[0, 1] + \
			0.5 * sin(2 * yps) * cos(phi) * deltas[2, 0] + \
			0.5 * sin(2 * yps) * sin(phi) * deltas[1, 2]

		B = cos(yps) * sin(2 * phi) * (0.5 * deltas[0, 0] - 0.5 * deltas[1, 1]) - \
			cos(yps) * cos(2 * phi) * deltas[0, 1] - \
			sin(yps) * sin(phi) * deltas[2, 0] + \
			sin(yps) * cos(phi) * deltas[1, 2]

		Sperp_squared = \
			0.5 * (cos(yps) ** 2 * cos(phi) ** 2 + sin(phi) ** 2) * 0.5 * deltas[0, 0] + \
			0.5 * (cos(yps) ** 2 * sin(phi) ** 2 + cos(phi) ** 2) * 0.5 * deltas[1, 1] + \
			0.5 * sin(yps) ** 2 * 0.5 * deltas[2, 2] - \
			0.25 * sin(yps) ** 2 * sin(2 * phi) * deltas[0, 1] - \
			0.25 * sin(2 * yps) * cos(phi) * deltas[2, 0] - \
			0.25 * sin(2 * yps) * sin(phi) * deltas[1, 2] - \
			0.5 * numpy.sqrt(A ** 2 + B ** 2)

		Na = n1.mean()
		Nb = n2.mean()

		return (Na + Nb) * Sperp_squared / (S ** 2)
