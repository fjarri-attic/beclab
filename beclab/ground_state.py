"""
Ground state calculation classes
"""

import copy
import numpy

from .helpers import *
from .wavefunction import WavefunctionSet
from .meters import ParticleStatistics
from .constants import *


class Projector(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid

		mask = getProjectorMask(constants, grid)
		if int(mask.sum()) == mask.size:
			self.is_identity = True
		else:
			self.is_identity = False
			self._projector_mask = env.toDevice(mask)

		self._addParameters(components=2, ensembles=1)

	def _prepare(self):
		self._p.comp_msize = self._p.ensembles * self._grid.msize

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void projector(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *projector_mask)
			{
				LIMITED_BY(${p.comp_msize});
				SCALAR mask_val = projector_mask[GLOBAL_INDEX % ${g.msize}];
				COMPLEX val;

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + ${comp * p.comp_msize}];
				data[GLOBAL_INDEX + ${comp * p.comp_msize}] = complex_mul_scalar(val, mask_val);
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)
		self._kernel_projector = self._program.projector

	def _cpu__kernel_projector(self, gsize, data, projector_mask):
		mask = numpy.tile(projector_mask,
			(self._p.components, self._p.ensembles) + (1,) * self._grid.dim)

		data *= mask

	def __call__(self, data):
		if not self.is_identity:
			self._kernel_projector(self._p.comp_msize, data, self._projector_mask)


class TFGroundState(PairedCalculation):
	"""
	Ground state, calculated using Thomas-Fermi approximation
	(kinetic energy == 0)
	"""

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid
		self._potentials = env.toDevice(getPotentials(constants, grid))
		self._stats = ParticleStatistics(env, constants, grid)
		self._projector = Projector(env, constants, grid)

		if isinstance(grid, HarmonicGrid):
			self._plan = createFHTPlan(env, constants, grid, 1)

		self._addParameters(components=2)
		self.prepare()

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=1)
		self._stats.prepare(components=self._p.components)
		self._p.g = numpy.array([
			self._constants.g[c, c] / self._constants.hbar for c in xrange(self._p.components)
		])

	def _gpu__prepare_specific(self):
		kernel_template = """
			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			EXPORTED_FUNC void fillWithTFGroundState(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM SCALAR *potentials, SCALAR mu0_by_hbar, SCALAR mu1_by_hbar)
			{
				LIMITED_BY(gsize);

				SCALAR potential = potentials[GLOBAL_INDEX];
				SCALAR e;

				%for comp in xrange(p.components):
				e = mu${comp}_by_hbar - potential;
				res[GLOBAL_INDEX + gsize * ${comp}] =
					e > 0 ?
						complex_ctr(sqrt(e / (SCALAR)${p.g[comp]}), 0) :
						complex_ctr(0, 0);
				%endfor
			}

			EXPORTED_FUNC void multiplyConstantsCS(int gsize, GLOBAL_MEM COMPLEX *data,
				SCALAR coeff0, SCALAR coeff1)
			{
				LIMITED_BY(gsize);
				COMPLEX val;

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + gsize * ${comp}];
				data[GLOBAL_INDEX + gsize * ${comp}] =
					complex_mul_scalar(val, coeff${comp});
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)
		self._kernel_fillWithTFGroundState = self._program.fillWithTFGroundState
		self._kernel_multiplyConstantsCS = self._program.multiplyConstantsCS

	def _cpu__kernel_fillWithTFGroundState(self, gsize, data, potentials,
			mu0_by_hbar, mu1_by_hbar):
		mask_func = lambda x: 0.0 if x < 0 else x
		mask_map = numpy.vectorize(mask_func)
		mu = (mu0_by_hbar, mu1_by_hbar)

		for c in xrange(self._p.components):
			data[c, 0] = numpy.sqrt(mask_map(mu[c] - self._potentials) / self._p.g[c])

	def _cpu__kernel_multiplyConstantsCS(self, gsize, data, coeff0, coeff1):
		coeffs = (coeff0, coeff1)
		for comp in xrange(self._p.components):
			data[comp] *= coeffs[comp]

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
		self._projector(psi.data)
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
		N_target = numpy.array(N)
		N_real = self._stats.getN(psi)

		coeffs = []
		for target, real in zip(N_target, N_real):
			coeffs.append(numpy.sqrt(target / real) if real != 0 else 1.0)

		cast = self._constants.scalar.cast

		# TODO: generalize for components > 2 if necessary
		self._kernel_multiplyConstantsCS(self._grid.size, psi.data,
			cast(coeffs[0]), cast(coeffs[1] if len(coeffs) > 1 else 1))
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
		self._constants = constants
		self._grid = grid
		self._tf_gs = TFGroundState(env, constants, grid)
		self._statistics = ParticleStatistics(env, constants, grid)
		self._addParameters(components=1)

	def _prepare(self):
		self._tf_gs.prepare(components=self._p.components)
		self._statistics.prepare(components=self._p.components)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void multiplyConstantCS(int gsize, GLOBAL_MEM COMPLEX *data,
				SCALAR c0, SCALAR c1)
			{
				LIMITED_BY(gsize);
				COMPLEX val;

				%for component in xrange(p.components):
				val = data[GLOBAL_INDEX + gsize * ${component}];
				data[GLOBAL_INDEX + gsize * ${component}] =
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
		self._potentials = env.toDevice(getPotentials(constants, grid))
		self._addParameters(dt=1e-5, itmax=3, precision=1e-6)
		self._projector = Projector(env, constants, grid)
		self.prepare(**kwds)

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=1)
		self._p.relative_precision = self._p.precision / self._p.dt
		self._p.g = self._constants.g / self._constants.hbar
		energy = self._grid.energy
		self._mode_prop = self._env.toDevice(numpy.exp(energy * (-self._p.dt / 2)))

	def _gpu__prepare_specific(self, **kwds):
		kernel_template = """
			// Propagates psi function in mode space
			EXPORTED_FUNC void mpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *mode_prop)
			{
				LIMITED_BY(gsize);
				SCALAR prop = mode_prop[GLOBAL_INDEX];
				COMPLEX val;

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + gsize * ${comp}];
				data[GLOBAL_INDEX + gsize * ${comp}] = complex_mul_scalar(val, prop);
				%endfor
			}

			// Propagates state in x-space for steady state calculation
			EXPORTED_FUNC void xpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials)
			{
				LIMITED_BY(gsize);

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + gsize * ${comp}];
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
				data[GLOBAL_INDEX + gsize * ${comp}] =
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
		self._projector(psi.data)
		return self._p.dt


class RK5Propagation(PairedCalculation):

	def __init__(self, env, constants, grid, mspace):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self._maxFinder = createMaxFinder(self._env, self._constants.complex.dtype)

		self._addParameters(dt_guess=1e-7, eps=1e-6, tiny=0, mspace=mspace,
			components=1, ensembles=1)

	def _prepare(self):

		self._p.dt = self._p.dt_guess
		self._err_old = 1.0

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

		if self._p.mspace:
			shape = self._grid.mshape
			self._p.grid_size = self._grid.msize
		else:
			shape = self._grid.shape
			self._p.grid_size = self._grid.size

		self._p.comp_size = self._p.grid_size * self._p.ensembles
		shape = (self._p.components, self._p.ensembles) + shape

		cdtype = self._constants.complex.dtype
		sdtype = self._constants.scalar.dtype

		self._maxFinder.prepare(length=self._grid.size * self._p.components * self._p.ensembles)
		self._c_maxbuffer = self._env.allocate((1,), dtype=cdtype)

		self._xdata = self._env.allocate(shape, dtype=cdtype)
		self._k = self._env.allocate((6,) + shape, dtype=cdtype)
		self._scale = self._env.allocate(shape, dtype=cdtype)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void calculateScale(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY(${p.comp_size});
				COMPLEX deriv, val;
				int id;

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + ${comp * p.comp_size};
				deriv = k[id];
				val = data[id];
				res[id] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${p.tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${p.tiny}
				);
				%endfor
			}

			EXPORTED_FUNC void calculateError(int gsize, GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *scale)
			{
				LIMITED_BY(${p.comp_size});
				COMPLEX val, s;
				int id;

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + ${comp * p.comp_size};
				val = k[id];
				s = scale[id];
				k[id] = complex_ctr(
					val.x / s.x / (SCALAR)${p.eps},
					val.y / s.y / (SCALAR)${p.eps}
				);
				%endfor
			}

			EXPORTED_FUNC void createData(int gsize,
				GLOBAL_MEM COMPLEX *res, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *k, int stage)
			{
				LIMITED_BY(${p.comp_size});
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
				val = data[GLOBAL_INDEX + ${comp * p.comp_size}];
				for(int s = 0; s < stage; s++)
				{
					kval = k[GLOBAL_INDEX + s * ${p.comp_size * p.components} +
						${p.comp_size * comp}];
					val = val + complex_mul_scalar(kval, b[stage][s]);
				}

				res[GLOBAL_INDEX + ${comp * p.comp_size}] = val;
				%endfor
			}

			EXPORTED_FUNC void sumResults(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY(${p.comp_size});
				COMPLEX res_val, err_val, kval;

				%for comp in xrange(p.components):
				res_val = data[GLOBAL_INDEX + ${comp * p.comp_size}];
				err_val = complex_ctr(0, 0);
				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${p.comp_size * p.components * s} +
						${p.comp_size * comp}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${p.cval[s] - p.cerr[s]});
				%endfor
				res[GLOBAL_INDEX + ${p.comp_size * comp}] = res_val;
				k[GLOBAL_INDEX + ${p.comp_size * comp}] = complex_ctr(abs(err_val.x), abs(err_val.y));
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_calculateScale = self._program.calculateScale
		self._kernel_calculateError = self._program.calculateError
		self._kernel_createData = self._program.createData
		self._kernel_sumResults = self._program.sumResults

	def _cpu__kernel_calculateScale(self, gsize, res, k, data):
		for c in xrange(self._p.components):
			res[c].flat[:] = (
				numpy.abs(k[0, c].real) + 1j * numpy.abs(k[0, c].imag) +
				numpy.abs(data[c].real) + 1j * numpy.abs(data[c].imag) +
				(1 + 1j) * self._p.tiny).flat

	def _cpu__kernel_calculateError(self, gsize, k, scale):
		shape = (self._p.components, self._p.ensembles) + k.shape[3:]
		scale = scale.reshape(shape)
		k[0].real /= scale.real * self._p.eps
		k[0].imag /= scale.imag * self._p.eps

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

	def _propagate_rk5(self, prop, psi, dt0):

		cast = self._constants.scalar.cast
		for stage in xrange(6):
			self._kernel_createData(psi.size, self._xdata,
				psi.data, self._k, numpy.int32(stage))
			dt = self._p.a[stage] * dt0
			prop(self._k, self._xdata, dt, dt0, stage)

		self._kernel_sumResults(psi.size, self._xdata,
			self._k, psi.data)

	def propagate(self, prop, finalize, psi, max_dt=None):

		safety = 0.9
		eps = self._p.eps

		dt = self._p.dt
		if max_dt is not None and max_dt < dt:
			dt = max_dt
		cast = self._constants.scalar.cast

		# Estimate scale for this step
		prop(self._k, psi.data, 0.0, dt, 0)
		self._kernel_calculateScale(psi.size, self._scale, self._k, psi.data)

		# parameters for error controller
		k = 4 # order of the method (order - 1 if using dt * psi' in scale)
		beta = 0.4 / k
		alpha = 1.0 / k - 0.75 * beta
		minscale = 0.1
		maxscale = 5.0

		# Propagate
		while True:
			#print "Trying with step " + str(dt)
			self._propagate_rk5(prop, psi, dt)
			self._kernel_calculateError(psi.size, self._k, self._scale)
			self._maxFinder(self._k, self._c_maxbuffer)
			cmax = self._env.fromDevice(self._c_maxbuffer)
			errmax = max(abs(cmax.real[0]), abs(cmax.imag[0]))

			#print "Error: " + str(errmax)
			if errmax < 1.0:
			#	print "Seems ok"
				break

			# reducing step size and retrying step
			dt = max(safety * errmax ** (-alpha), minscale) * dt

		dt_used = dt

		if errmax == 0.0:
			scale = maxscale
		else:
			scale = safety * errmax ** (-alpha) * self._err_old ** beta
			if scale < minscale:
				scale = minscale
			if scale > maxscale:
				scale = maxscale
		self._err_old = errmax
		self._p.dt = dt * scale

		self._env.copyBuffer(self._xdata, dest=psi.data)
		finalize(psi, dt_used)
		return dt_used


class RK5IPGroundState(ImaginaryTimeGroundState):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)

		self._plan = createFFTPlan(env, constants, grid)
		self._potentials = env.toDevice(getPotentials(constants, grid))
		self._energy = env.toDevice(grid.energy)

		self._propagator = RK5Propagation(env, constants, grid, mspace=False)
		self._projector = Projector(env, constants, grid)

		self._addParameters(relative_precision=1e-0, atol_coeff=1e-3,
			eps=1e-6, dt_guess=1e-7, Nscale=10000)
		self.prepare(**kwds)

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=1)
		self._p.g = self._constants.g / self._constants.hbar

		mu = self._constants.muTF(self._p.Nscale, dim=self._grid.dim, comp=0)
		peak_density = numpy.sqrt(mu / self._constants.g[0, 0])

		# FIXME: toIP() can be done inplace in self._k,
		# we just need to support offsets in FFT (because I only want to transform part of it)
		self._buffer = self._env.allocate(
			(self._p.components, 1) + self._grid.shape,
			self._constants.complex.dtype
		)

		self._propagator.prepare(eps=self._p.eps, dt_guess=self._p.dt_guess, mspace=False,
			tiny=peak_density * self._p.atol_coeff, components=self._p.components)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void transformIP(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY(gsize);
				COMPLEX val;
				int id;
				SCALAR e = energy[GLOBAL_INDEX];

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + gsize * ${comp};
				val = data[id];
				data[id] = complex_mul_scalar(val, exp(e * dt));
				%endfor
			}

			EXPORTED_FUNC void propagationFunc(int gsize,
				GLOBAL_MEM COMPLEX *result, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0)
			{
				LIMITED_BY(gsize);
				SCALAR p = potentials[GLOBAL_INDEX];

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + gsize * ${comp}];
				SCALAR n${comp} = squared_abs(val${comp});
				%endfor

				%for comp in xrange(p.components):
				result[GLOBAL_INDEX + gsize * ${comp}] =
					complex_mul_scalar(val${comp}, -dt0 * (p
						%for comp_other in xrange(p.components):
						+ n${comp_other} * (SCALAR)${p.g[comp, comp_other]}
						%endfor
					));
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_transformIP = self._program.transformIP
		self._kernel_propagationFunc = self._program.propagationFunc

	def _cpu__kernel_transformIP(self, gsize, data, energy, dt):
		for c in xrange(self._p.components):
			data[c] *= numpy.exp(energy * dt)

	def _cpu__kernel_propagationFunc(self, gsize, result, data, potentials, dt0):
		g = self._p.g
		n = numpy.abs(data) ** 2

		tile = (self._p.components,) + (1,) * self._grid.dim
		p_tiled = numpy.tile(potentials, tile)

		result.flat[:] = p_tiled.flat
		for c in xrange(self._p.components):
			for other_c in xrange(self._p.components):
				result[c] += n[other_c, 0] * g[c, other_c]
			result[c] *= -dt0 * data[c, 0]

	def _propFunc(self, results, values, dt, dt_full, stage):
		self._fromIP(values, dt, False)
		self._kernel_propagationFunc(self._grid.size, self._buffer, values,
			self._potentials, self._constants.scalar.cast(dt_full))
		self._toIP(self._buffer, dt, True)
		self._env.copyBuffer(self._buffer, dest=results,
			dest_offset=stage * self._grid.size * self._p.components)

	def _finalizeFunc(self, psi, dt_used):
		self._fromIP(psi.data, dt_used, False)

	def _propagate(self, psi):
		dt_used = self._propagator.propagate(self._propFunc, self._finalizeFunc, psi)
		# FIXME: In an ideal world, it should not be necessary, because projection happens in toIP()
		# but it seems that the error is accumulated after every sumResults in RK propagator
		if not self._projector.is_identity:
			psi.toMSpace()
			self._projector(psi.data)
			psi.toXSpace()
		return dt_used

	def _toIP(self, data, dt, project):
		self._plan.execute(data, batch=self._p.components)
		if dt != 0.0:
			self._kernel_transformIP(self._grid.size, data, self._energy,
				self._constants.scalar.cast(dt))
		if project:
			self._projector(data)
		self._plan.execute(data, batch=self._p.components, inverse=True)

	def _fromIP(self, data, dt, project):
		self._toIP(data, -dt, project)

	def create(self, N, **kwds):
		kwds['Nscale'] = max(N)
		return ImaginaryTimeGroundState.create(self, N, **kwds)


class RK5HarmonicGroundState(ImaginaryTimeGroundState):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, HarmonicGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)

		self._energy = env.toDevice(grid.energy)
		self._plan3 = createFHTPlan(env, constants, grid, 3)

		self._projector = Projector(env, constants, grid)
		self._propagator = RK5Propagation(env, constants, grid, mspace=True)

		self._addParameters(kwds, relative_precision=1e-0,
			atol_coeff=1e-3, eps=1e-6, dt_guess=1e-7, Nscale=10000)

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=1)
		self._p.g = self._constants.g / self._constants.hbar

		shape = self._grid.mshape
		cdtype = self._constants.complex.dtype

		self._x3data = self._env.allocate((self._p.components, 1) + self._grid.shapes[3], dtype=cdtype)
		self._mdata = self._env.allocate((self._p.components, 1) + self._grid.mshape, dtype=cdtype)

		self._propagator.prepare(eps=self._p.eps, dt_guess=self._p.dt_guess,
			tiny=numpy.sqrt(self._p.Nscale) * self._p.atol_coeff, components=self._p.components)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void calculateNonlinear(int gsize, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY(gsize);

				SCALAR res;

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + gsize * ${comp}];
				SCALAR n${comp} = squared_abs(val${comp});
				%endfor

				%for comp in xrange(p.components):
				res =
					%for comp_other in xrange(p.components):
					- n${comp_other} * (SCALAR)${p.g[comp, comp_other]}
					%endfor
					;
				data[GLOBAL_INDEX + gsize * ${comp}] = complex_mul_scalar(val${comp}, res);
				%endfor
			}

			EXPORTED_FUNC void propagationFunc(int gsize,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *nldata, GLOBAL_MEM SCALAR *energy, SCALAR dt0, int stage)
			{
				LIMITED_BY(gsize);
				COMPLEX val, nlval;
				SCALAR e = energy[GLOBAL_INDEX];

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + gsize * ${comp}];
				nlval = nldata[GLOBAL_INDEX + gsize * ${comp}];
				k[GLOBAL_INDEX + gsize * ${p.components} * stage + gsize * ${comp}] =
					complex_mul_scalar(
						complex_mul_scalar(val, -e) + nlval,
						dt0);
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_propagationFunc = self._program.propagationFunc
		self._kernel_calculateNonlinear = self._program.calculateNonlinear

	def _cpu__kernel_calculateNonlinear(self, gsize, data):
		g = self._p.g
		n = numpy.abs(data) ** 2

		for comp in xrange(self._p.components):
			res = numpy.zeros_like(data[0])
			for comp_other in xrange(self._p.components):
				res -= n[comp_other] * g[comp, comp_other]
			data[comp] *= res

	def _cpu__kernel_propagationFunc(self, gsize, k, data, nldata, energy, dt0, stage):
		tile = (self._p.components,) + (1,) * (self._grid.dim + 1)
		e = numpy.tile(energy, tile)
		k[stage].flat[:] = ((-data * e + nldata) * dt0).flat

	def _propFunc(self, results, values, dt, dt_full, stage):
		cast = self._constants.scalar.cast
		self._plan3.execute(values, self._x3data, inverse=True, batch=self._p.components)
		self._kernel_calculateNonlinear(self._grid.sizes[3], self._x3data)
		self._plan3.execute(self._x3data, self._mdata, batch=self._p.components)
		self._projector(self._mdata)
		self._kernel_propagationFunc(self._grid.msize, results, values,
			self._mdata, self._energy, cast(dt_full), numpy.int32(stage))

	def _finalizeFunc(self, psi, dt_used):
		pass

	def _propagate(self, psi):
		dt_used = self._propagator.propagate(self._propFunc, self._finalizeFunc, psi)
		# FIXME: In an ideal world, it should not be necessary, because projection happens in toIP()
		# but it seems that the error is accumulated after every M->X->M transformation
		if not self._projector.is_identity:
			self._projector(psi.data)
		return dt_used

	def _toMeasurementSpace(self, psi):
		psi.toXSpace()

	def _toEvolutionSpace(self, psi):
		psi.toMSpace()
