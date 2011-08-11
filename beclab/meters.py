import numpy

from .helpers import *
from .constants import *


class ParticleStatistics(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self._potentials = getPotentials(env, constants, grid)

		if isinstance(grid, HarmonicGrid):
			self._energy = getHarmonicEnergy(env, constants, grid)
		elif isinstance(grid, UniformGrid):
			self._energy = getPlaneWaveEnergy(env, constants, grid)

		self._dV = grid.get_dV(env)

		self._sreduce_ensembles = createReduce(env, constants.scalar.dtype)
		self._sreduce_all = createReduce(env, constants.scalar.dtype)
		self._creduce_all = createReduce(env, constants.complex.dtype)
		self._sreduce_single_to_comps = createReduce(env, constants.scalar.dtype)

		self._density_modifiers = getDensityModifiers(env, constants, grid)
		self._zero_density_modifiers = self._env.toDevice(
			numpy.zeros(grid.shape).astype(constants.scalar.dtype))

		self._addParameters(components=2, ensembles=1)
		self.prepare(**kwds)

	def _prepare(self):
		self._p.g = self._constants.g / self._constants.hbar
		self._p.need_potentials = isinstance(self._grid, UniformGrid)

		self._sreduce_ensembles.prepare(
			sparse=True,
			batch=self._p.components,
			final_length=self._grid.size,
			length=self._p.ensembles * self._grid.size)

		self._sreduce_all.prepare(
			sparse=False, final_length=self._p.components,
			length=self._p.components * self._p.ensembles * self._grid.size)

		self._sreduce_single_to_comps.prepare(
			sparse=False, final_length=self._p.components,
			length=self._p.components * self._grid.size)

		self._creduce_all.prepare(
			sparse=False, final_length=1,
			length=self._p.ensembles * self._grid.size)

		self._c_mspace_buffer = self._env.allocate(
			(self._p.components, self._p.ensembles) + self._grid.mshape,
			self._constants.complex.dtype)
		self._c_xspace_buffer = self._env.allocate(
			(self._p.components, self._p.ensembles) + self._grid.shape,
			self._constants.complex.dtype)
		self._s_xspace_buffer = self._env.allocate(
			(self._p.components, self._p.ensembles) + self._grid.shape,
			self._constants.scalar.dtype)
		self._s_xspace_buffer_single = self._env.allocate(
			(self._p.components, 1) + self._grid.shape,
			self._constants.scalar.dtype)
		self._s_comp_buffer = self._env.allocate(
			(self._p.components,),
			self._constants.scalar.dtype)
		self._c_1_buffer = self._env.allocate(
			(1,),
			self._constants.complex.dtype)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void interaction(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY(gsize);
				COMPLEX val0 = data[GLOBAL_INDEX + gsize * ${0}];
				COMPLEX val1 = data[GLOBAL_INDEX + gsize * ${1}];

				res[GLOBAL_INDEX] = complex_mul(val0, conj(val1));
			}

			EXPORTED_FUNC void density(int gsize, GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *state, GLOBAL_MEM SCALAR *modifiers, int coeff)
			{
				LIMITED_BY(gsize);
				int id;
				SCALAR modifier = modifiers[GLOBAL_INDEX % (gsize / ${p.ensembles})];

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + gsize * ${comp};
				res[id] = (squared_abs(state[id]) - modifier) / coeff;
				%endfor
			}

			EXPORTED_FUNC void invariant(int gsize, GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *xdata, GLOBAL_MEM COMPLEX *mdata,
				GLOBAL_MEM SCALAR *potentials, int coeff)
			{
				LIMITED_BY(gsize);

				%if p.need_potentials:
				SCALAR potential = potentials[GLOBAL_INDEX % (gsize / ${p.ensembles})];
				%endif

				%for comp in xrange(p.components):
				int id${comp} = GLOBAL_INDEX + gsize * ${comp};
				SCALAR n${comp} = squared_abs(xdata[id${comp}]);
				%endfor

				%for comp in xrange(p.components):
				SCALAR nonlinear${comp} = ${'potential' if p.need_potentials else '0'};
					%for comp_other in xrange(p.components):
					nonlinear${comp} += (SCALAR)${p.g[comp, comp_other]} * n${comp_other} / coeff;
					%endfor
				nonlinear${comp} *= n${comp};
				COMPLEX differential${comp} = complex_mul(
					conj(xdata[id${comp}]), mdata[id${comp}]);

				// differential.y will be equal to 0, because \psi * D \psi is a real number
				res[id${comp}] = nonlinear${comp} + differential${comp}.x;
				%endfor
			}

			EXPORTED_FUNC void multiplyTiledSS(int gsize,
				GLOBAL_MEM SCALAR *data, GLOBAL_MEM SCALAR *coeffs, int ensembles)
			{
				LIMITED_BY(gsize);

				SCALAR coeff_val = coeffs[GLOBAL_INDEX % (gsize / ensembles)];
				SCALAR data_val;

				%for comp in xrange(p.components):
				data_val = data[GLOBAL_INDEX + gsize * ${comp}];
				data[GLOBAL_INDEX + gsize * ${comp}] = data_val * coeff_val;
				%endfor
			}

			EXPORTED_FUNC void multiplyTiledCS(int gsize,
				GLOBAL_MEM COMPLEX *data, GLOBAL_MEM SCALAR *coeffs, int components)
			{
				LIMITED_BY(gsize);

				SCALAR coeff_val = coeffs[GLOBAL_INDEX % (gsize / ${p.ensembles})];
				COMPLEX data_val;

				for(int comp = 0; comp < components; comp++)
				{
					data_val = data[GLOBAL_INDEX + gsize * comp];
					data[GLOBAL_INDEX + gsize * comp] =
						complex_mul_scalar(data_val, coeff_val);
				}
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_interaction = self._program.interaction
		self._kernel_invariant = self._program.invariant
		self._kernel_density = self._program.density
		self._kernel_multiplyTiledSS = self._program.multiplyTiledSS
		self._kernel_multiplyTiledCS = self._program.multiplyTiledCS

	def _cpu__kernel_interaction(self, gsize, res, data):
		self._env.copyBuffer(data[0] * data[1].conj(), dest=res)

	def _cpu__kernel_density(self, gsize, density, data, modifiers, coeff):
		modifiers = numpy.tile(modifiers,
			(self._p.components, self._p.ensembles,) + (1,) * self._grid.dim)
		self._env.copyBuffer((numpy.abs(data) ** 2 - modifiers) / coeff, dest=density)

	def _cpu__kernel_multiplyTiledSS(self, gsize, data, coeffs, ensembles):
		data *= numpy.tile(coeffs,
			(self._p.components, ensembles,) + (1,) * self._grid.dim)

	def _cpu__kernel_multiplyTiledCS(self, gsize, data, coeffs, components):
		data *= numpy.tile(coeffs,
			(components, self._p.ensembles,) + (1,) * self._grid.dim)

	def _cpu__kernel_invariant(self, gsize, res, xdata, mdata, potentials, coeff):

		tile = (self._p.ensembles,) + (1,) * self._grid.dim
		g = self._p.g
		n = numpy.abs(xdata) ** 2
		components = self._p.components

		self._env.copyBuffer(numpy.zeros_like(res), dest=res)
		if self._p.need_potentials:
			for comp in xrange(components):
				res[comp] += numpy.tile(potentials, tile) * n[comp]

		for comp in xrange(components):
			for comp_other in xrange(components):
				res[comp] += n[comp] * (g[comp, comp_other] * n[comp_other] / coeff)
			res[comp] += (xdata[comp].conj() * mdata[comp]).real

	def getVisibility(self, psi):
		assert self._p.components == 2
		N = self.getN(psi)
		self._kernel_interaction(psi.size, self._c_xspace_buffer, psi.data)
		self._kernel_multiplyTiledCS(psi.size, self._c_xspace_buffer,
			self._dV, numpy.int32(1))
		self._creduce_all(self._c_xspace_buffer, self._c_1_buffer)
		interaction = self._env.fromDevice(self._c_1_buffer)
		interaction = numpy.abs(interaction[0]) / self._p.ensembles

		return 2.0 * interaction / N.sum()

	def getDensity(self, psi, coeff=1):
		if psi.type == WIGNER:
			modifiers = self._density_modifiers
		else:
			modifiers = self._zero_density_modifiers

		self._kernel_density(psi.size, self._s_xspace_buffer, psi.data,
			modifiers, numpy.int32(coeff))
		return self._s_xspace_buffer

	def getAverageDensity(self, psi):
		# Using psi.size and .shape instead of grid here, to make it work
		# for both x- and mode-space.
		ensembles = psi.ensembles
		components = psi.components
		size = self._grid.msize if psi.in_mspace else self._grid.size

		density = self.getDensity(psi, coeff=ensembles)
		self._sreduce_ensembles(density, self._s_xspace_buffer_single)

		return self._s_xspace_buffer_single

	def getAveragePopulation(self, psi):
		density = self.getAverageDensity(psi)
		if not psi.in_mspace:
			self._kernel_multiplyTiledSS(self._grid.size, density, self._dV, numpy.int32(1))
		return density

	def _getInvariant(self, psi, coeff, N):

		# TODO: work out the correct formula for Wigner function's E/mu
		if psi.type != CLASSICAL:
			raise NotImplementedError()

		# If N is not known beforehand, we have to calculate it first
		if N is None:
			N = self.getN(psi).sum()

		batch = self._p.ensembles * self._p.components
		xsize = self._grid.size * self._p.ensembles
		msize = self._grid.msize * self._p.ensembles

		# FIXME: not a good way to provide transformation
		psi._plan.execute(psi.data, self._c_mspace_buffer, batch=batch)
		self._kernel_multiplyTiledCS(msize, self._c_mspace_buffer, self._energy,
			numpy.int32(self._p.components))
		psi._plan.execute(self._c_mspace_buffer, self._c_xspace_buffer,
			batch=batch, inverse=True)
		cast = self._constants.scalar.cast

		self._kernel_invariant(xsize, self._s_xspace_buffer,
			psi.data, self._c_xspace_buffer,
			self._potentials, numpy.int32(coeff))
		self._kernel_multiplyTiledSS(xsize, self._s_xspace_buffer, self._dV,
			numpy.int32(self._p.ensembles))

		self._sreduce_all(self._s_xspace_buffer, self._s_comp_buffer)
		comps = self._env.fromDevice(self._s_comp_buffer)

		return comps / self._p.ensembles / N * self._constants.hbar

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
		"""Returns particle count for wavefunction"""
		p = self.getAveragePopulation(psi)
		self._sreduce_single_to_comps(p, self._s_comp_buffer)
		comps = self._env.fromDevice(self._s_comp_buffer)
		return comps

	def getEnergy(self, psi, N=None):
		"""Returns average energy per particle"""
		return self._getInvariant(psi, 2, N)

	def getMu(self, psi, N=None):
		"""Returns average chemical potential per particle"""
		return self._getInvariant(psi, 1, N)


class DensityProfile(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()
		self._stats = ParticleStatistics(env, constants, grid)
		self._addParameters(components=2, ensembles=1)
		self._reduce = createReduce(env, constants.scalar.dtype)
		self.prepare()

	def _prepare(self):
		self._stats.prepare(components=self._p.components, ensembles=self._p.ensembles)
		self._reduce.prepare(length=self._p.components * self._grid.size,
			final_length=self._grid.shape[0] * self._p.components)
		self._z_buffer = self._env.allocate(
			(self._p.components, self._grid.shape[0]), self._constants.scalar.dtype)

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
		self._reduce(p, self._z_buffer)
		res = self._env.fromDevice(self._z_buffer)
		return res / numpy.tile(self._grid.dz, (self._p.components, 1))


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
