"""
Ground state calculation classes
"""

import copy
import numpy

from .helpers import *
from .wavefunction import WavefunctionSet
from .meters import ParticleStatistics
from .constants import getPotentials, getPlaneWaveEnergy, getHarmonicEnergy, UniformGrid, HarmonicGrid


class TFGroundState(PairedCalculation):
	"""
	Ground state, calculated using Thomas-Fermi approximation
	(kinetic energy == 0)
	"""

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()
		self._potentials = getPotentials(env, constants, grid)
		self._stats = ParticleStatistics(env, constants, grid)

		if isinstance(grid, HarmonicGrid):
			self._plan = createFHTPlan(env, constants, grid, 1)

		self._addParameters(components=1)
		self.prepare()

	def _prepare(self):
		self._stats.prepare(components=self._p.components)
		self._p.g = numpy.array([
			self._constants.g[c, c] / self._constants.hbar for c in xrange(self._p.components)
		])

	def _gpu__prepare_specific(self):
		kernel_template = """
			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			EXPORTED_FUNC void fillWithTFGroundState(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM SCALAR *potentials, SCALAR mu0_by_hbar, SCALAR mu1_by_hbar)
			{
				LIMITED_BY_GRID;

				SCALAR potential = potentials[GLOBAL_INDEX];
				SCALAR e;

				%for comp in xrange(p.components):
				e = mu${comp}_by_hbar - potential;
				res[GLOBAL_INDEX + ${g.size * comp}] =
					e > 0 ?
						complex_ctr(sqrt(e / (SCALAR)${p.g[comp]}), 0) :
						complex_ctr(0, 0);
				%endfor
			}

			EXPORTED_FUNC void multiplyConstantCS(GLOBAL_MEM COMPLEX *data, SCALAR coeff)
			{
				LIMITED_BY_GRID;
				COMPLEX val;

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + ${g.size * comp}];
				data[GLOBAL_INDEX + ${g.size * comp}] =
					complex_mul_scalar(val, coeff);
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)
		self._kernel_fillWithTFGroundState = self._program.fillWithTFGroundState
		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS

	def _cpu__kernel_fillWithTFGroundState(self, gsize, data, potentials,
			mu0_by_hbar, mu1_by_hbar):
		mask_func = lambda x: 0.0 if x < 0 else x
		mask_map = numpy.vectorize(mask_func)
		mu = (mu0_by_hbar, mu1_by_hbar)

		for c in xrange(self._p.components):
			data[c, 0] = numpy.sqrt(mask_map(mu[c] - self._potentials) / self._p.g[c])

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, coeff):
		data *= coeff

	def fillWithTF(self, psi, N):
		mu_by_hbar = numpy.array([
			self._constants.muTF(N[i], dim=self._grid.dim, comp=i) if i < len(N) else 0
			for i in xrange(2)
		]).astype(self._constants.scalar.dtype) / self._constants.hbar

		# TODO: generalize for components > 2 if necessary
		self._kernel_fillWithTFGroundState(psi.size, psi.data,
			self._potentials, mu_by_hbar[0], mu_by_hbar[1])

		# This is required for HarmonicGrid
		# Otherwise first X-M-X transform removes some "excessive" parts
		# TODO: need to mathematically justify this
		psi.toMSpace()
		psi.toXSpace()

		# The total number of atoms is equal to the number requested
		# only in the limit of infinite number of lattice points.
		# So we have to renormalize the data.
		#
		# We can do it in x- or mode-space.
		# For uniform grid it does not matter. For harmonic grid there will be
		# some difference (see comment in getHarmonicGrid() in fht.py).
		# Doing it in x-space because all losses, interaction and noise are
		# calculated in x-space, and kinetic + potential operator is less significant.
		#psi.toMSpace()
		N_target = numpy.array(N).sum()
		N_real = self._stats.getN(psi).sum()
		coeff = numpy.sqrt(N_target / N_real)
		self._kernel_multiplyConstantCS(psi.size, psi.data, self._constants.scalar.cast(coeff))
		#psi.toXSpace()

	def create(self, N):
		psi = WavefunctionSet(self._env, self._constants, self._grid, components=len(N))

		if isinstance(N, int):
			N = (N,)
		assert len(N) <= 2
		self.prepare(components=len(N))

		self.fillWithTF(psi, N)
		return psi


class ImaginaryTimeGroundState(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()
		self._tf_gs = TFGroundState(env, constants, grid)
		self._statistics = ParticleStatistics(env, constants, grid)
		self._addParameters(components=1)

	def _prepare(self):
		self._tf_gs.prepare(components=self._p.components)
		self._statistics.prepare(components=self._p.components)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void multiplyConstantCS(GLOBAL_MEM COMPLEX *data,
				SCALAR c0, SCALAR c1)
			{
				LIMITED_BY_GRID;
				COMPLEX val;

				%for component in xrange(p.components):
				val = data[GLOBAL_INDEX + ${g.size * component}];
				data[GLOBAL_INDEX + ${g.size * component}] =
					complex_mul_scalar(val, c${component});
				%endfor
			}
		"""

		self.__program = self.compileProgram(kernel_template)
		self._kernel_multiplyConstantCS = self.__program.multiplyConstantCS

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, c0, c1):
		coeffs = (c0, c1)
		for c in xrange(self._p.components):
			data[c] *= coeffs[c]

	def _renormalize(self, psi, coeffs):
		cast = self._constants.scalar.cast
		c0 = coeffs[0]
		c1 = coeffs[1] if self._p.components > 1 else 0
		self._kernel_multiplyConstantCS(psi.size, psi.data, cast(c0), cast(c1))

	def _toEvolutionSpace(self, psi):
		pass

	def _toMeasurementSpace(self, psi):
		pass

	def _create(self, psi, N):

		# it would be nice to use two-component TF state here,
		# but the formula is quite complex, and it is much easier
		# just to start from something approximately correct
		self._tf_gs.fillWithTF(psi, N)
		N_target = numpy.array(N).sum()

		stats = self._statistics
		precision = self._p.relative_precision
		dt_used = 0

		total_N = lambda psi: stats.getN(psi).sum()
		total_E = lambda psi, N: stats.getEnergy(psi, N=N).sum()
		total_mu = lambda psi, N: stats.getMu(psi, N=N).sum()

		E = 0.0

		new_E = total_E(psi, N_target)

		self._toEvolutionSpace(psi)

		# Reducing the dependence on time step
		# Now we can use small time steps not being afraid that
		# propagation will be terminated too soon (because dE is too small)
		# (TODO: dE ~ dt, but not exactly; see W. Bao and Q. Du, 2004, eqn. 2.7
		# Now default precision is chosen so that usual dt's work well with it)
		while abs(E - new_E) / new_E > precision * dt_used:

			# propagation
			dt_used = self._propagate(psi)

			# renormalization
			self._toMeasurementSpace(psi)
			new_N = stats.getN(psi)
			coeffs = [numpy.sqrt(N[c] / new_N[c]) for c in xrange(self._p.components)]
			self._renormalize(psi, coeffs)

			E = new_E
			new_E = total_E(psi, N_target)
			self._toEvolutionSpace(psi)

			if new_E > E:
				print "!!! Warning: energy started to rise, propagation aborted."
				break

		self._toMeasurementSpace(psi)

	def create(self, N, **kwds):
		if isinstance(N, int):
			N = (N,)
		assert len(N) <= 2

		if len(N) == 1:
			psi = WavefunctionSet(self._env, self._constants, self._grid, components=1)
			self.prepare(components=1, **kwds)
			self._create(psi, N)
		elif len(N) == 2 and N[1] == 0:
			psi1 = WavefunctionSet(self._env, self._constants, self._grid, components=1)
			self.prepare(components=1, **kwds)
			self._create(psi1, (N[0],))
			psi = WavefunctionSet(self._env, self._constants, self._grid, components=2)
			psi.fillComponent(0, psi1, 0)
		else:
			psi = WavefunctionSet(self._env, self._constants, self._grid, components=2)
			self.prepare(components=2, **kwds)
			self._create(psi, N)

		return psi


class SplitStepGroundState(ImaginaryTimeGroundState):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)
		self._potentials = getPotentials(env, constants, grid)
		self._addParameters(dt=1e-5, itmax=3, precision=1e-6)
		self.prepare(**kwds)

	def _prepare(self):
		self._p.relative_precision = self._p.precision / self._p.dt
		self._p.g = self._constants.g / self._constants.hbar
		energy = getPlaneWaveEnergy(None, self._constants, self._grid)
		self._mode_prop = self._env.toDevice(numpy.exp(energy * (-self._p.dt / 2)))

	def _gpu__prepare_specific(self, **kwds):
		kernel_template = """
			// Propagates psi function in mode space
			EXPORTED_FUNC void mpropagate(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *mode_prop)
			{
				LIMITED_BY_GRID;
				SCALAR prop = mode_prop[GLOBAL_INDEX];
				COMPLEX val;

				%for component in xrange(p.components):
				val = data[GLOBAL_INDEX + ${g.size * component}];
				data[GLOBAL_INDEX + ${g.size * component}] = complex_mul_scalar(val, prop);
				%endfor
			}

			// Propagates state in x-space for steady state calculation
			EXPORTED_FUNC void xpropagate(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials)
			{
				LIMITED_BY_GRID;

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + ${g.size * comp}];
				COMPLEX val${comp}_copy = val${comp}; // store initial x-space field
				SCALAR dval${comp}, n${comp};
				%endfor

				SCALAR V = potentials[GLOBAL_INDEX];

				// iterate to midpoint solution
				%for i in range(p.itmax):
					// calculate midpoint log derivative and exponentiate
					%for comp in xrange(p.components):
					n${comp} = squared_abs(val${comp});
					%endfor

					%for comp in xrange(p.components):
					dval${comp} = exp((SCALAR)${p.dt / 2.0} * (-V
						%for other_comp in xrange(p.components):
						- (SCALAR)${p.g[comp, other_comp]} * n${other_comp}
						%endfor
					));

					//propagate to midpoint using log derivative
					val${comp} = complex_mul_scalar(val${comp}_copy, dval${comp});
					%endfor
				%endfor

				//propagate to endpoint using log derivative
				%for comp in xrange(p.components):
				data[GLOBAL_INDEX + ${g.size * comp}] =
					complex_mul_scalar(val${comp}, dval${comp});
				%endfor
			}
		"""

		self.__program = self.compileProgram(kernel_template)

		self._kernel_mpropagate = self.__program.mpropagate
		self._kernel_xpropagate = self.__program.xpropagate

	def _cpu__kernel_mpropagate(self, gsize, data, mode_prop):
		for c in xrange(self._p.components):
			data[c, 0] *= mode_prop

	def _cpu__kernel_xpropagate(self, gsize, data, potentials):
		data_copy = data.copy()
		g = self._p.g
		dt = -self._p.dt / 2
		tile = (self._p.components, 1,) + (1,) * self._grid.dim
		p_tiled = numpy.tile(potentials, tile)

		for i in xrange(self._p.itmax):
			n = numpy.abs(data) ** 2
			dp = p_tiled.copy()
			for c in xrange(self._p.components):
				for other_c in xrange(self._p.components):
					dp[c] += n[other_c] * g[c, other_c]

			d = numpy.exp(dp * dt)
			data.flat[:] = (data_copy * d).flat

		data *= d

	def _toEvolutionSpace(self, psi):
		psi.toMSpace()

	def _toMeasurementSpace(self, psi):
		psi.toXSpace()

	def _propagate(self, psi):
		self._kernel_mpropagate(psi.size, psi.data, self._mode_prop)
		self._toMeasurementSpace(psi)
		self._kernel_xpropagate(psi.size, psi.data, self._potentials)
		self._toEvolutionSpace(psi)
		self._kernel_mpropagate(psi.size, psi.data, self._mode_prop)
		return self._p.dt


class RK5IPGroundState(ImaginaryTimeGroundState):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)

		self._plan = createFFTPlan(self._env, self._constants, self._grid)
		self._potentials = getPotentials(self._env, self._constants, self._grid)
		self._energy = getPlaneWaveEnergy(self._env, self._constants, self._grid)
		self._maxFinder = createMaxFinder(self._env, self._constants.scalar.dtype)

		self._addParameters(dt_guess=1e-4, eps=1e-7, tiny=1e-4, relative_precision=1e-0)
		self.prepare(**kwds)

	def _prepare(self):
		self._p.g = self._constants.g / self._constants.hbar

		self._p.dt = self._p.dt_guess
		self._p.a = numpy.array([0, 0.2, 0.3, 0.6, 1, 0.875])
		self._p.b = numpy.array([
			[0, 0, 0, 0, 0],
			[1.0 / 5, 0, 0, 0, 0],
			[3.0 / 40, 9.0 / 40, 0, 0, 0],
			[3.0 / 10, -9.0 / 10, 6.0 / 5, 0, 0],
			[-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27, 0],
			[1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]
		])
		self._p.cval = numpy.array([37.0 / 378, 0, 250.0 / 621, 125.0 / 594, 0, 512.0 / 1771])
		self._p.cerr = numpy.array([2825.0 / 27648, 0, 18575.0 / 48384.0, 13525.0 / 55296, 277.0 / 14336, 0.25])

		shape = self._grid.mshape
		cdtype = self._constants.complex.dtype
		sdtype = self._constants.scalar.dtype

		self._xdata = self._env.allocate((self._p.components, 1) + shape, dtype=cdtype)
		self._k = self._env.allocate((6, self._p.components, 1) + shape, dtype=cdtype)
		self._scale = self._env.allocate((self._p.components, 1) + shape, dtype=cdtype)

	def _gpu__prepare_specific(self):
		ImaginaryTimeGroundState._gpu__prepare_specific(self)

		kernel_template = """
			EXPORTED_FUNC void transformIP(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY_GRID;
				COMPLEX val;
				int id;
				SCALAR e = energy[GLOBAL_INDEX];

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + ${comp * g.size};
				val = data[id];
				data[id] = complex_mul_scalar(val, exp(e * dt));
				%endfor
			}

			EXPORTED_FUNC void calculateScale(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX deriv, val;
				int id;

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + ${comp * g.size};
				deriv = k[id];
				val = data[id];
				res[id] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${p.tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${p.tiny}
				);
				%endfor
			}

			EXPORTED_FUNC void calculateError(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *scale)
			{
				LIMITED_BY_GRID;
				COMPLEX val, s;
				int id;

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + ${comp * g.size};
				val = k[id];
				s = scale[id];
				k[id] = complex_ctr(
					val.x / s.x / (SCALAR)${p.eps},
					val.y / s.y / (SCALAR)${p.eps}
				);
				%endfor
			}

			EXPORTED_FUNC void propagationFunc(GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0, int stage)
			{
				LIMITED_BY_GRID;
				SCALAR p = potentials[GLOBAL_INDEX];

				%for comp in xrange(p.components):
				COMPLEX val${comp};
				SCALAR n${comp};
				%endfor

				%for comp in xrange(p.components):
				val${comp} = data[GLOBAL_INDEX + ${g.size * comp}];
				n${comp} = squared_abs(val${comp});
				%endfor

				%for comp in xrange(p.components):
				k[GLOBAL_INDEX + ${g.size * p.components} * stage + ${g.size * comp}] =
					complex_mul_scalar(val${comp}, -dt0 * (p
						%for comp_other in xrange(p.components):
						+ n${comp_other} * (SCALAR)${p.g[comp, comp_other]}
						%endfor
					));
				%endfor
			}

			EXPORTED_FUNC void createData(GLOBAL_MEM COMPLEX *res, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *k, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val, kval;

				const SCALAR b[6][5] = {
					%for stage in xrange(6):
					{
						%for s in xrange(5):
						(SCALAR)${p.b[stage, s]},
						%endfor
					},
					%endfor
				};

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + ${g.size * comp}];
				for(int s = 0; s < stage; s++)
				{
					kval = k[GLOBAL_INDEX + s * ${g.size * p.components} + ${g.size * comp}];
					val = val + complex_mul_scalar(kval, b[stage][s]);
				}

				res[GLOBAL_INDEX + ${g.size * comp}] = val;
				%endfor
			}

			EXPORTED_FUNC void sumResults(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX res_val, err_val, kval;

				%for comp in xrange(p.components):
				res_val = data[GLOBAL_INDEX + ${g.size * comp}];
				err_val = complex_ctr(0, 0);
				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${g.size * p.components * s} + ${g.size * comp}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s] - p.cerr[s]});
				%endfor
				res[GLOBAL_INDEX + ${g.size * comp}] = res_val;
				k[GLOBAL_INDEX + ${g.size * comp}] = complex_ctr(abs(err_val.x), abs(err_val.y));
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_transformIP = self._program.transformIP
		self._kernel_calculateScale = self._program.calculateScale
		self._kernel_calculateError = self._program.calculateError
		self._kernel_propagationFunc = self._program.propagationFunc
		self._kernel_createData = self._program.createData
		self._kernel_sumResults = self._program.sumResults

	def _cpu__kernel_transformIP(self, gsize, data, energy, dt):
		for c in xrange(self._p.components):
			data[c] *= numpy.exp(energy * dt)

	def _cpu__kernel_calculateScale(self, gsize, res, k, data):
		for c in xrange(self._p.components):
			res[c].flat[:] = (
				numpy.abs(k[0, c].real) + 1j * numpy.abs(k[0, c].imag) +
				numpy.abs(data[c].real) + 1j * numpy.abs(data[c].imag) +
				(1 + 1j) * self._p.tiny).flat

	def _cpu__kernel_calculateError(self, gsize, k, scale):
		shape = k.shape[2:]
		scale = scale.reshape(shape)
		for c in xrange(self._p.components):
			k[0, c].real /= scale.real * self._p.eps
			k[0, c].imag /= scale.imag * self._p.eps

	def _cpu__kernel_propagationFunc(self, gsize, k, data, potentials, dt0, stage):
		g = self._p.g
		n = numpy.abs(data) ** 2

		tile = (self._p.components,) + (1,) * self._grid.dim
		p_tiled = numpy.tile(potentials, tile)

		k[stage] = p_tiled.copy()
		for c in xrange(self._p.components):
			for other_c in xrange(self._p.components):
				k[stage, c] += n[other_c, 0] * g[c, other_c]
			k[stage, c] *= -dt0 * data[c, 0]

	def _cpu__kernel_createData(self, gsize, res, data, k, stage):
		b = self._p.b[stage, :]
		for c in xrange(self._p.components):
			res[c].flat[:] = data[c].flat
			for s in xrange(stage):
 				res[c] += k[s, c] * b[s]

	def _cpu__kernel_sumResults(self, gsize, res, k, data):
		cval = self._p.cval
		cerr = cval - self._p.cerr

		for c in xrange(self._p.components):
			res[c].flat[:] = data[c].flat

			for s in xrange(6):
				res[c] += k[s, c] * cval[s]

			k[0, c] *= cerr[0]
			for s in xrange(1, 6):
				k[0, c] += k[s, c] * cerr[s]

			k[0, c] = numpy.abs(k[0, c].real) + 1j * numpy.abs(k[0, c].imag)

	def _propagate_rk5(self, psi, dt0):

		cast = self._constants.scalar.cast
		for stage in xrange(6):
			self._kernel_createData(psi.size, self._xdata,
				psi.data, self._k, numpy.int32(stage))
			dt = self._p.a[stage] * dt0
			self._fromIP(self._xdata, dt)
			self._kernel_propagationFunc(psi.size, self._k, self._xdata,
				self._potentials, cast(dt0), numpy.int32(stage))

		self._kernel_sumResults(psi.size, self._xdata,
			self._k, psi.data)

	def _propagate(self, psi):

		safety = 0.9
		eps = self._p.eps

		dt = self._p.dt
		cast = self._constants.scalar.cast

		# Estimate scale for this step
		self._kernel_propagationFunc(psi.size, self._k, psi.data, self._potentials,
				cast(dt), numpy.int32(0))
		self._kernel_calculateScale(psi.size, self._scale, self._k, psi.data)

		# Propagate

		while True:
			#print "Trying with step " + str(dt)
			self._propagate_rk5(psi, dt)
			self._kernel_calculateError(psi.size, self._k, self._scale)

			errmax = self._maxFinder(self._k, length=psi.size * self._p.components)
			#print "Error: " + str(errmax)
			if errmax < 1.0:
			#	if dt > remaining_time:
			#		# Step is fine in terms of error, but bigger then necessary
			#		dt = remaining_time
			#		continue
			#	else:
			#		#print "Seems ok"
			#		break
			#	print "Seems ok"
				break

			# reducing step size and retrying step
			dt_temp = safety * dt * (errmax ** (-0.25))
			dt = max(dt_temp, 0.1 * dt)

		dt_used = dt

		if errmax > (5.0 / safety) ** (-1.0 / 0.2):
			self._p.dt = safety * dt * (errmax ** (-0.2))
		else:
			self._p.dt = 5.0 * dt

		self._env.copyBuffer(self._xdata, dest=psi.data)
		self._fromIP(psi.data, dt_used)
		return dt_used

	def _toIP(self, data, dt):
		if dt == 0.0:
			return

		self._plan.execute(data, batch=self._p.components)
		self._kernel_transformIP(data.size, data, self._energy, self._constants.scalar.cast(dt))
		self._plan.execute(data, batch=self._p.components, inverse=True)

	def _fromIP(self, data, dt):
		self._toIP(data, -dt)


class RK5HarmonicGroundState(ImaginaryTimeGroundState):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, HarmonicGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)

		self._energy = getHarmonicEnergy(self._env, self._constants, self._grid)
		self._maxFinder = createMaxFinder(self._env, self._constants.scalar.dtype)
		self._plan3 = createFHTPlan(env, constants, grid, 3)

		shape = self._grid.mshape
		cdtype = self._constants.complex.dtype
		sdtype = self._constants.scalar.dtype

		self._xdata0 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._xdata1 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._x3data0 = self._env.allocate((1,) + grid.shapes[3], dtype=cdtype)

		self._k = self._env.allocate((2, 6) + shape, dtype=cdtype)

		self._scale0 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._scale1 = self._env.allocate((1,) + shape, dtype=cdtype)

		self._initParameters(kwds, comp0=0, comp1=1, dt_guess=1e-4, eps=1e-7, tiny=1e-6,
			relative_precision=1e-0)

	def _prepare(self):
		ImaginaryTimeGroundState._prepare(self)

		g_by_hbar = self._constants.g / self._constants.hbar
		self._p.g00 = g_by_hbar[self._p.comp0, self._p.comp0]
		self._p.g01 = g_by_hbar[self._p.comp0, self._p.comp1]
		self._p.g11 = g_by_hbar[self._p.comp1, self._p.comp1]

		self._p.dt = self._p.dt_guess
		self._p.a = numpy.array([0, 0.2, 0.3, 0.6, 1, 0.875])
		self._p.b = numpy.array([
			[0, 0, 0, 0, 0],
			[1.0 / 5, 0, 0, 0, 0],
			[3.0 / 40, 9.0 / 40, 0, 0, 0],
			[3.0 / 10, -9.0 / 10, 6.0 / 5, 0, 0],
			[-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27, 0],
			[1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]
		])
		self._p.cval = numpy.array([37.0 / 378, 0, 250.0 / 621, 125.0 / 594, 0, 512.0 / 1771])
		self._p.cerr = numpy.array([2825.0 / 27648, 0, 18575.0 / 48384.0, 13525.0 / 55296, 277.0 / 14336, 0.25])

	def _gpu__prepare_specific(self):
		ImaginaryTimeGroundState._gpu__prepare_specific(self)

		kernel_template = """
			EXPORTED_FUNC void calculateScale(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX deriv = k[GLOBAL_INDEX];
				COMPLEX val = data[GLOBAL_INDEX];
				res[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${p.tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${p.tiny}
				);
			}

			EXPORTED_FUNC void calculateScale_2comp(GLOBAL_MEM COMPLEX *res0,
				GLOBAL_MEM COMPLEX *res1,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1)
			{
				LIMITED_BY_GRID;
				COMPLEX deriv, val;

				deriv = k[GLOBAL_INDEX];
				val = data0[GLOBAL_INDEX];
				res0[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${p.tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${p.tiny}
				);

				deriv = k[GLOBAL_INDEX + ${g.size}];
				val = data1[GLOBAL_INDEX];
				res1[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${p.tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${p.tiny}
				);
			}

			EXPORTED_FUNC void calculateError(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *scale)
			{
				LIMITED_BY_GRID;
				COMPLEX val = k[GLOBAL_INDEX];
				COMPLEX s = scale[GLOBAL_INDEX];
				k[GLOBAL_INDEX] = complex_ctr(
					val.x / s.x / (SCALAR)${p.eps},
					val.y / s.y / (SCALAR)${p.eps}
				);
			}

			EXPORTED_FUNC void calculateError_2comp(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *scale0, GLOBAL_MEM COMPLEX *scale1)
			{
				LIMITED_BY_GRID;
				COMPLEX val, s;

				val = k[GLOBAL_INDEX];
				s = scale0[GLOBAL_INDEX];
				k[GLOBAL_INDEX] = complex_ctr(
					val.x / s.x / (SCALAR)${p.eps},
					val.y / s.y / (SCALAR)${p.eps}
				);

				val = k[GLOBAL_INDEX + ${g.size}];
				s = scale1[GLOBAL_INDEX];
				k[GLOBAL_INDEX + ${g.size}] = complex_ctr(
					val.x / s.x / (SCALAR)${p.eps},
					val.y / s.y / (SCALAR)${p.eps}
				);
			}

			EXPORTED_FUNC void calculateNonlinear(GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				SCALAR n = squared_abs(val);
				data[GLOBAL_INDEX] = complex_mul_scalar(val, -n * (SCALAR)${p.g00});
			}

			EXPORTED_FUNC void propagationFunc(GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *nldata, GLOBAL_MEM SCALAR *energy, SCALAR dt0, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				COMPLEX nlval = nldata[GLOBAL_INDEX];
				SCALAR e = energy[GLOBAL_INDEX];
				k[GLOBAL_INDEX + ${g.size} * stage] = complex_mul_scalar(
					complex_mul_scalar(val, -e) + nlval,
					dt0);
			}

			EXPORTED_FUNC void createData(GLOBAL_MEM COMPLEX *res, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *k, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				COMPLEX kval;

				const SCALAR b[6][5] = {
					%for stage in xrange(6):
					{
						%for s in xrange(5):
						(SCALAR)${p.b[stage, s]},
						%endfor
					},
					%endfor
				};

				for(int s = 0; s < stage; s++)
				{
					kval = k[GLOBAL_INDEX + s * ${g.size}];
					val = val + complex_mul_scalar(kval, b[stage][s]);
				}

				res[GLOBAL_INDEX] = val;
			}

			EXPORTED_FUNC void createData_2comp(GLOBAL_MEM COMPLEX *res0,
				GLOBAL_MEM COMPLEX *res1, GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, GLOBAL_MEM COMPLEX *k, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val0 = data0[GLOBAL_INDEX];
				COMPLEX val1 = data1[GLOBAL_INDEX];
				COMPLEX kval;
				SCALAR bval;

				const SCALAR b[6][5] = {
					%for stage in xrange(6):
					{
						%for s in xrange(5):
						(SCALAR)${p.b[stage, s]},
						%endfor
					},
					%endfor
				};

				for(int s = 0; s < stage; s++)
				{
					bval = b[stage][s];
					kval = k[GLOBAL_INDEX + s * ${g.size}];
					val0 = val0 + complex_mul_scalar(kval, bval);

					kval = k[GLOBAL_INDEX + s * ${g.size} + ${g.size * 6}];
					val1 = val1 + complex_mul_scalar(kval, bval);
				}

				res0[GLOBAL_INDEX] = val0;
				res1[GLOBAL_INDEX] = val1;
			}

			EXPORTED_FUNC void sumResults(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX res_val = data[GLOBAL_INDEX];
				COMPLEX err_val = complex_ctr(0, 0);
				COMPLEX kval;

				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${g.size * s}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s] - p.cerr[s]});
				%endfor
				res[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX] = complex_ctr(abs(err_val.x), abs(err_val.y));
			}

			EXPORTED_FUNC void sumResults_2comp(GLOBAL_MEM COMPLEX *res0,
				GLOBAL_MEM COMPLEX *res1,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1)
			{
				LIMITED_BY_GRID;
				COMPLEX res_val;
				COMPLEX err_val;
				COMPLEX kval;

				res_val = data0[GLOBAL_INDEX];
				err_val = complex_ctr(0, 0);
				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${g.size * s}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s] - p.cerr[s]});
				%endfor
				res0[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX] = complex_ctr(abs(err_val.x), abs(err_val.y));

				res_val = data1[GLOBAL_INDEX];
				err_val = complex_ctr(0, 0);
				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${g.size * s + g.size * 6}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s] - p.cerr[s]});
				%endfor
				res1[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX + ${g.size}] = complex_ctr(abs(err_val.x), abs(err_val.y));
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, p=self._p)

		self._kernel_calculateScale = self._program.calculateScale
		self._kernel_calculateError = self._program.calculateError
		self._kernel_propagationFunc = self._program.propagationFunc
		self._kernel_calculateNonlinear = self._program.calculateNonlinear
		self._kernel_createData = self._program.createData
		self._kernel_sumResults = self._program.sumResults

	def _cpu__kernel_calculateScale(self, gsize, res, k, data):
		res.flat[:] = (
			numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag) +
			numpy.abs(data.real) + 1j * numpy.abs(data.imag) +
			(1 + 1j) * self._p.tiny).flat

	def _cpu__kernel_calculateScale_2comp(self, gsize, res0, res1, k, data0, data1):
		res0.flat[:] = (
			numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag) +
			numpy.abs(data0.real) + 1j * numpy.abs(data0.imag) +
			(1 + 1j) * self._p.tiny).flat

		res1.flat[:] = (
			numpy.abs(k[1, 0].real) + 1j * numpy.abs(k[1, 0].imag) +
			numpy.abs(data1.real) + 1j * numpy.abs(data1.imag) +
			(1 + 1j) * self._p.tiny).flat

	def _cpu__kernel_calculateError(self, gsize, k, scale):
		shape = k.shape[2:]
		scale = scale.reshape(shape)
		k[0, 0].real /= scale.real * self._p.eps
		k[0, 0].imag /= scale.imag * self._p.eps

	def _cpu__kernel_calculateNonlinear(self, gsize, data):
		g = self._p.g00
		n = numpy.abs(data) ** 2
		data *= -(n * g)

	def _cpu__kernel_propagationFunc(self, gsize, k, data, nldata, energy, dt0, stage):
		k[0, stage] = (-data * energy + nldata) * dt0

	def _cpu__kernel_createData(self, gsize, res, data, k, stage):
		res.flat[:] = data.flat

		b = self._p.b[stage, :]
		for s in xrange(stage):
 			res += k[0, s] * b[s]

	def _cpu__kernel_createData_2comp(self, gsize, res0, res1, data0, data1, k, stage):
		res0.flat[:] = data0.flat
		res1.flat[:] = data1.flat

		b = self._p.b[stage, :]
		for s in xrange(stage):
 			res0 += k[0, s] * b[s]
			res1 += k[1, s] * b[s]

	def _cpu__kernel_sumResults(self, gsize, res, k, data):
		res.flat[:] = data.flat

		c = self._p.cval
		c_err = c - self._p.cerr

		for s in xrange(6):
			res += k[0, s] * c[s]

		k[0, 0] *= c_err[0]
		for s in xrange(1, 6):
			k[0, 0] += k[0, s] * c_err[s]
		k[0, 0] = numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag)

	def _cpu__kernel_sumResults_2comp(self, gsize, res0, res1, k, data0, data1):
		res0.flat[:] = data0.flat
		res1.flat[:] = data1.flat

		c = self._p.cval
		c_err = c - self._p.cerr

		for s in xrange(6):
			res0 += k[0, s] * c[s]
			res1 += k[0, s] * c[s]

		k[0, 0] *= c_err[0]
		k[1, 0] *= c_err[0]
		for s in xrange(1, 6):
			k[0, 0] += k[0, s] * c_err[s]
			k[1, 0] += k[1, s] * c_err[s]

	def _propagate_rk5(self, psi0, dt0):

		cast = self._constants.scalar.cast

		for stage in xrange(6):
			self._kernel_createData(psi0.size, self._xdata0,
				psi0.data, self._k, numpy.int32(stage))
			dt = self._p.a[stage] * dt0
			self._plan3.execute(self._xdata0, self._x3data0, inverse=True)
			self._kernel_calculateNonlinear(self._x3data0.size, self._x3data0)
			self._plan3.execute(self._x3data0, self._xdata1)
			self._kernel_propagationFunc(psi0.size, self._k, self._xdata0,
				self._xdata1, self._energy, cast(dt0), numpy.int32(stage))

		self._kernel_sumResults(psi0.size, self._xdata0,
			self._k, psi0.data)

	def _propagate_rk5_2comp(self, psi0, psi1, dt0):

		cast = self._constants.scalar.cast

		for stage in xrange(6):
			self._kernel_createData_2comp(psi0.size, self._xdata0, self._xdata1,
				psi0.data, psi1.data, self._k, numpy.int32(stage))
			dt = self._p.a[stage] * dt0
			self._fromIP(self._xdata0, self._xdata1, dt)
			self._kernel_propagationFunc_2comp(psi0.size, self._k, self._xdata0, self._xdata1,
				self._potentials, cast(dt0), numpy.int32(stage))
			self._toIP(self._xdata0, self._xdata1, dt)

		self._kernel_sumResults_2comp(psi0.size, self._xdata0, self._xdata1,
			self._k, psi0.data, psi1.data)

	def _propagate(self, psi0, psi1):

		safety = 0.9
		eps = self._p.eps

		dt = self._p.dt
		cast = self._constants.scalar.cast

		# Estimate scale for this step

		if psi1 is None:
			self._plan3.execute(psi0.data, self._x3data0, inverse=True)
			self._kernel_calculateNonlinear(self._x3data0.size, self._x3data0)
			self._plan3.execute(self._x3data0, self._xdata1)
			self._kernel_propagationFunc(psi0.size, self._k, psi0.data,
				self._xdata1, self._energy, cast(dt), numpy.int32(0))
			self._kernel_calculateScale(psi0.size, self._scale0,
				self._k, psi0.data)
		else:
			self._kernel_propagationFunc_2comp(psi0.size, self._k, psi0.data, psi1.data,
				self._potentials, cast(dt), numpy.int32(0))
			self._kernel_calculateScale_2comp(psi0.size, self._scale0, self._scale1,
				self._k, psi0.data, psi1.data)

		# Propagate

		while True:
			#print "Trying with step " + str(dt)
			if psi1 is None:
				self._propagate_rk5(psi0, dt)
				self._kernel_calculateError(psi0.size, self._k, self._scale0)
				errmax = self._maxFinder(self._k, length=psi0.size)
			else:
				self._propagate_rk5_2comp(psi0, psi1, dt)
				self._kernel_calculateError_2comp(psi0.size, self._k, self._scale0, self._scale1)
				errmax = self._maxFinder(self._k, length=psi0.size * 2)

			#print "Error: " + str(errmax)
			if errmax < 1.0:
			#	if dt > remaining_time:
			#		# Step is fine in terms of error, but bigger then necessary
			#		dt = remaining_time
			#		continue
			#	else:
			#		#print "Seems ok"
			#		break
			#	print "Seems ok"
				break

			# reducing step size and retrying step
			dt_temp = safety * dt * (errmax ** (-0.25))
			dt = max(dt_temp, 0.1 * dt)

		dt_used = dt

		if errmax > (5.0 / safety) ** (-1.0 / 0.2):
			self._p.dt = safety * dt * (errmax ** (-0.2))
		else:
			self._p.dt = 5.0 * dt

		self._env.copyBuffer(self._xdata0, dest=psi0.data)
		if psi1 is not None:
			self._env.copyBuffer(self._xdata1, dest=psi1.data)

		return dt_used

	def _toMeasurementSpace(self, psi0, psi1):
		psi0.toXSpace()
		if psi1 is not None:
			psi1.toXSpace()

	def _toEvolutionSpace(self, psi0, psi1):
		psi0.toMSpace()
		if psi1 is not None:
			psi1.toMSpace()
