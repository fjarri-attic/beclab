"""
Ground state calculation classes
"""

import copy
import numpy

from .helpers import *
from .wavefunction import Wavefunction, TwoComponentCloud
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

		self._initParameters()

	def _gpu__prepare_specific(self):
		kernel_template = """
			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			EXPORTED_FUNC void fillWithTFGroundState(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM SCALAR *potentials, SCALAR mu_by_hbar,
				SCALAR g_by_hbar)
			{
				LIMITED_BY_GRID;

				SCALAR potential = potentials[GLOBAL_INDEX];

				SCALAR e = mu_by_hbar - potential;
				if(e > 0)
					res[GLOBAL_INDEX] = complex_ctr(sqrt(e / g_by_hbar), 0);
				else
					res[GLOBAL_INDEX] = complex_ctr(0, 0);
			}

			EXPORTED_FUNC void multiplyConstantCS(GLOBAL_MEM COMPLEX *data, SCALAR coeff)
			{
				LIMITED_BY_GRID;
				COMPLEX x = data[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(x, coeff);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants, self._grid)
		self._kernel_fillWithTFGroundState = self._program.fillWithTFGroundState
		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS

	def _cpu__kernel_fillWithTFGroundState(self, gsize, data, potentials, mu_by_hbar, g_by_hbar):
		mask_func = lambda x: 0.0 if x < 0 else x
		mask_map = numpy.vectorize(mask_func)
		self._env.copyBuffer(
			numpy.sqrt(mask_map(mu_by_hbar - self._potentials) / g_by_hbar),
			dest=data)

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, coeff):
		data *= coeff

	def _fillWithTF(self, data, g, mu):
		cast = self._constants.scalar.cast
		mu_by_hbar = cast(mu / self._constants.hbar)
		g_by_hbar = cast(g / self._constants.hbar)

		self._kernel_fillWithTFGroundState(data.size, data,
			self._potentials, mu_by_hbar, g_by_hbar)

	def fillWithTF(self, psi, N):
		comp = psi.comp
		g = self._constants.g[comp, comp]
		mu = self._constants.muTF(N, dim=self._grid.dim, comp=comp)
		self._fillWithTF(psi.data, g, mu)

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
		N_real = self._stats.getN(psi)
		coeff = numpy.sqrt(N / N_real)
		self._kernel_multiplyConstantCS(psi.size, psi.data, self._constants.scalar.cast(coeff))
		#psi.toXSpace()

	def create(self, N, comp=0):
		psi = Wavefunction(self._env, self._constants, self._grid, comp=comp)
		self.fillWithTF(psi, N)
		return psi

	def createCloud(self, N, ratio=1.0):
		cloud = TwoComponentCloud(self._env, self._constants, self._grid)
		self.fillWithTF(cloud.psi0, N * ratio)
		if ratio != 1.0:
			self.fillWithTF(cloud.psi1, N * (1.0 - ratio))
		return cloud


class ImaginaryTimeGroundState(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()
		self._tf_gs = TFGroundState(env, constants, grid)
		self._statistics = ParticleStatistics(env, constants, grid)

	def _prepare(self):
		#self._statistics.prepare()
		pass

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void multiplyConstantCS(GLOBAL_MEM COMPLEX *data, SCALAR c)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(val, c);
			}

			EXPORTED_FUNC void multiplyConstantCS_2comp(GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, SCALAR c0, SCALAR c1)
			{
				LIMITED_BY_GRID;

				COMPLEX d;
				d = data0[GLOBAL_INDEX];
				data0[GLOBAL_INDEX] = complex_mul_scalar(d, c0);
				d = data1[GLOBAL_INDEX];
				data1[GLOBAL_INDEX] = complex_mul_scalar(d, c1);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, p=self._p)

		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS
		self._kernel_multiplyConstantCS_2comp = self._program.multiplyConstantCS_2comp

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, c):
		data *= c

	def _cpu__kernel_multiplyConstantCS_2comp(self, gsize, data0, data1, c0, c1):
		data0 *= c0
		data1 *= c1

	def _renormalize(self, psi0, psi1, c0, c1):
		cast = self._constants.scalar.cast
		if psi1 is None:
			self._kernel_multiplyConstantCS(psi0.size, psi0.data, cast(c0))
		else:
			self._kernel_multiplyConstantCS_2comp(psi0.size,
				psi0.data, psi1.data, cast(c0), cast(c1))

	def _toEvolutionSpace(self, psi0, psi1):
		pass

	def _toMeasurementSpace(self, psi0, psi1):
		pass

	def _create(self, psi0, psi1, N0, N1):

		two_component = psi1 is not None
		verbose = False

		# it would be nice to use two-component TF state here,
		# but the formula is quite complex, and it is much easier
		# just to start from something approximately correct
		self._tf_gs.fillWithTF(psi0, N0)
		if two_component:
			self._tf_gs.fillWithTF(psi1, N1)

		stats = self._statistics
		precision = self._p.relative_precision
		dt_used = self._p.dt

		if two_component:
			total_N = lambda psi0, psi1: stats.getN(psi0) + stats.getN(psi1)
			total_E = lambda psi0, psi1, N: stats.getEnergy2comp(psi0, psi1, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu2comp(psi0, psi1, N=N)
		else:
			total_N = lambda psi0, psi1: stats.getN(psi0)
			total_E = lambda psi0, psi1, N: stats.getEnergy(psi0, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu(psi0, N=N)

		E = 0.0

		new_E = total_E(psi0, psi1, N0 + N1)
		self._toEvolutionSpace(psi0, psi1)

		# Reducing the dependence on time step
		# Now we can use small time steps not being afraid that
		# propagation will be terminated too soon (because dE is too small)
		# (TODO: dE ~ dt, but not exactly; see W. Bao and Q. Du, 2004, eqn. 2.7
		# Now default precision is chosen so that usual dt's work well with it)
		while abs(E - new_E) / new_E > precision * dt_used:

			# propagation
			dt_used = self._propagate(psi0, psi1)

			# renormalization
			self._toMeasurementSpace(psi0, psi1)
			if two_component:
				new_N0 = stats.getN(psi0)
				new_N1 = stats.getN(psi1)
				c0 = numpy.sqrt(N0 / new_N0)
				c1 = numpy.sqrt(N1 / new_N1)
				self._renormalize(psi0, psi1, c0, c1)
			else:
				new_N0 = stats.getN(psi0)
				self._renormalize(psi0, psi1, numpy.sqrt(N0 / new_N0), None)

			E = new_E
			new_E = total_E(psi0, psi1, N0 + N1)
			self._toEvolutionSpace(psi0, psi1)

		self._toMeasurementSpace(psi0, psi1)

		if verbose:
			postfix = "(two components)" if two_component else "(one component)"
			pop = str(stats.getN(psi0)) + " + " + str(stats.getN(psi1)) if two_component else \
				str(stats.getN(psi0))

			print "Ground state calculation " + postfix + " :" + \
					" N = " + N + \
					" E = " + str(total_E(psi0, psi1, N0 + N1)) + \
					" mu = " + str(total_mu(psi0, psi1, N0 + N1))

	def create(self, N, comp, verbose=False, **kwds):
		psi = Wavefunction(self._env, self._constants, self._grid, comp=comp)
		self.prepare(comp0=comp, **kwds)
		self._create(psi, None, N, 0)
		return psi

	def createCloud(self, N, ratio=1.0, verbose=False, **kwds):
		cloud = TwoComponentCloud(self._env, self._constants, self._grid)
		self.prepare(comp0=cloud.psi0.comp, comp1=cloud.psi1.comp, **kwds)
		if ratio == 1.0:
			self._create(cloud.psi0, None, N, 0)
		else:
			self._create(cloud.psi0, cloud.psi1, N * ratio, N * (1.0 - ratio))
		return cloud


class SplitStepGroundState(ImaginaryTimeGroundState):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)
		self._potentials = getPotentials(env, constants, grid)
		self._initParameters(kwds, dt=1e-5, comp0=0, comp1=1, itmax=3, precision=1e-2)

	def _prepare(self):
		ImaginaryTimeGroundState._prepare(self)

		g_by_hbar = self._constants.g / self._constants.hbar
		self._p.g00 = g_by_hbar[self._p.comp0, self._p.comp0]
		self._p.g01 = g_by_hbar[self._p.comp0, self._p.comp1]
		self._p.g11 = g_by_hbar[self._p.comp1, self._p.comp1]

		energy = getPlaneWaveEnergy(None, self._constants, self._grid)
		self._mode_prop = self._env.toDevice(numpy.exp(energy * (-self._p.dt / 2)))

	def _gpu__prepare_specific(self, **kwds):
		ImaginaryTimeGroundState._gpu__prepare_specific(self)

		kernel_template = """
			// Propagates psi function in mode space
			EXPORTED_FUNC void mpropagate(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *mode_prop)
			{
				LIMITED_BY_GRID;

				SCALAR prop = mode_prop[GLOBAL_INDEX];
				COMPLEX val = data[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(val, prop);
			}

			// Propagates two psi functions in mode space
			EXPORTED_FUNC void mpropagate_2comp(
				GLOBAL_MEM COMPLEX *data0, GLOBAL_MEM COMPLEX *data1,
				GLOBAL_MEM SCALAR *mode_prop)
			{
				LIMITED_BY_GRID;

				COMPLEX val;
				SCALAR prop = mode_prop[GLOBAL_INDEX];
				val = data0[GLOBAL_INDEX];
				data0[GLOBAL_INDEX] = complex_mul_scalar(val, prop);
				val = data1[GLOBAL_INDEX];
				data1[GLOBAL_INDEX] = complex_mul_scalar(val, prop);
			}

			// Propagates state in x-space for steady state calculation
			EXPORTED_FUNC void xpropagate(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials)
			{
				LIMITED_BY_GRID;

				COMPLEX val = data[GLOBAL_INDEX];

				// store initial x-space field
				COMPLEX val_copy = val;

				SCALAR dval;
				SCALAR V = potentials[GLOBAL_INDEX];

				// iterate to midpoint solution
				%for i in range(p.itmax):
					// calculate midpoint log derivative and exponentiate
					dval = exp((SCALAR)${p.dt / 2.0} *
						(-V - (SCALAR)${p.g00} * squared_abs(val)));

					//propagate to midpoint using log derivative
					val = complex_mul_scalar(val_copy, dval);
				%endfor

				//propagate to endpoint using log derivative
				data[GLOBAL_INDEX] = complex_mul_scalar(val, dval);
			}

			// Propagates state in x-space for steady state calculation
			EXPORTED_FUNC void xpropagate_2comp(GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, GLOBAL_MEM SCALAR *potentials)
			{
				LIMITED_BY_GRID;

				COMPLEX val0 = data0[GLOBAL_INDEX];
				COMPLEX val1 = data1[GLOBAL_INDEX];

				// store initial x-space field
				COMPLEX val0_copy = val0;
				COMPLEX val1_copy = val1;

				SCALAR dval0, dval1, n0, n1;
				SCALAR V = potentials[GLOBAL_INDEX];

				// iterate to midpoint solution
				%for i in range(p.itmax):
					// calculate midpoint log derivative and exponentiate
					n0 = squared_abs(val0);
					n1 = squared_abs(val1);

					dval0 = exp((SCALAR)${p.dt / 2.0} *
						(-V - (SCALAR)${p.g00} * n0 - (SCALAR)${p.g01} * n1));
					dval1 = exp((SCALAR)${p.dt / 2.0} *
						(-V - (SCALAR)${p.g01} * n0 - (SCALAR)${p.g11} * n1));

					// propagate to midpoint using log derivative
					val0 = complex_mul_scalar(val0_copy, dval0);
					val1 = complex_mul_scalar(val1_copy, dval1);
				%endfor

				// propagate to endpoint using log derivative
				data0[GLOBAL_INDEX] = complex_mul_scalar(val0, dval0);
				data1[GLOBAL_INDEX] = complex_mul_scalar(val1, dval1);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, p=self._p)

		self._kernel_mpropagate = self._program.mpropagate
		self._kernel_mpropagate_2comp = self._program.mpropagate_2comp
		self._kernel_xpropagate = self._program.xpropagate
		self._kernel_xpropagate_2comp = self._program.xpropagate_2comp

	def _cpu__kernel_mpropagate(self, gsize, data0, mode_prop):
		data0 *= mode_prop

	def _cpu__kernel_mpropagate_2comp(self, gsize, data0, data1, mode_prop):
		data0 *= mode_prop
		data1 *= mode_prop

	def _cpu__kernel_xpropagate(self, gsize, data, potentials):
		data_copy = data.copy()
		g = self._p.g00
		dt = -self._p.dt / 2

		for i in xrange(self._p.itmax):
			n = numpy.abs(data) ** 2
			d = numpy.exp((potentials + n * g) * dt)
			data.flat[:] = (data_copy * d).flat
		data *= d

	def _cpu__kernel_xpropagate_2comp(self, gsize, data0, data1, potentials):

		dt = -self._p.dt / 2
		g00 = self._p.g00
		g01 = self._p.g01
		g11 = self._p.g11

		data0_copy = data0.copy()
		data1_copy = data1.copy()

		for i in xrange(self._p.itmax):
			n0 = numpy.abs(data0) ** 2
			n1 = numpy.abs(data1) ** 2

			d0 = numpy.exp((potentials + n0 * g00 + n1 * g01) * dt)
			d1 = numpy.exp((potentials + n0 * g01 + n1 * g11) * dt)

			data0.flat[:] = (data0_copy * d0).flat
			data1.flat[:] = (data1_copy * d1).flat

		data0 *= d0
		data1 *= d1

	def _mpropagate(self, psi0, psi1):
		if psi1 is None:
			self._kernel_mpropagate(psi0.size, psi0.data, self._mode_prop)
		else:
			self._kernel_mpropagate_2comp(psi0.size, psi0.data, psi1.data, self._mode_prop)

	def _xpropagate(self, psi0, psi1):
		if psi1 is None:
			self._kernel_xpropagate(psi0.size, psi0.data, self._potentials)
		else:
			self._propagateXSpace2(psi0.size, psi0.data, psi1.data, self._potentials)

	def _toEvolutionSpace(self, psi0, psi1):
		psi0.toMSpace()
		if psi1 is not None:
			psi1.toMSpace()

	def _toMeasurementSpace(self, psi0, psi1):
		psi0.toXSpace()
		if psi1 is not None:
			psi1.toXSpace()

	def _propagate(self, psi0, psi1):
		self._mpropagate(psi0, psi1)
		self._toMeasurementSpace(psi0, psi1)
		self._xpropagate(psi0, psi1)
		self._toEvolutionSpace(psi0, psi1)
		self._mpropagate(psi0, psi1)
		return self._p.dt


class RK5IPGroundState(PairedCalculation):

	def __init__(self, env, constants, grid, dt_guess=1e-4, eps=1e-9):
		PairedCalculation.__init__(self, env)

		assert isinstance(grid, UniformGrid)

		self._constants = constants.copy()
		self._grid = grid.copy()

		self._dt = dt_guess
		self._eps = eps

		self._tf_gs = TFGroundState(env, constants, grid)
		self._statistics = ParticleStatistics(env, constants, grid)

		self._plan = createFFTPlan(self._env, self._constants, self._grid)
		self._potentials = getPotentials(self._env, self._constants, self._grid)
		self._energy = getPlaneWaveEnergy(self._env, self._constants, self._grid)
		self._maxFinder = createMaxFinder(self._env, self._constants.scalar.dtype)

		shape = self._grid.mshape
		cdtype = self._constants.complex.dtype
		sdtype = self._constants.scalar.dtype

		self._xdata0 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._xdata1 = self._env.allocate((1,) + shape, dtype=cdtype)

		self._k = self._env.allocate((2, 6) + shape, dtype=cdtype)

		self._scale0 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._scale1 = self._env.allocate((1,) + shape, dtype=cdtype)

	def _prepare(self, g00=0.0, g01=0.0, g11=0.0):
		self._dt_used = 0

		self._g00 = g00
		self._g01 = g01
		self._g11 = g11

		self._a = numpy.array([0, 0.2, 0.3, 0.6, 1, 0.875])
		self._b = numpy.array([
			[0, 0, 0, 0, 0],
			[1.0 / 5, 0, 0, 0, 0],
			[3.0 / 40, 9.0 / 40, 0, 0, 0],
			[3.0 / 10, -9.0 / 10, 6.0 / 5, 0, 0],
			[-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27, 0],
			[1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]
		])
		self._c = numpy.array([37.0 / 378, 0, 250.0 / 621, 125.0 / 594, 0, 512.0 / 1771])
		self._cs = numpy.array([2825.0 / 27648, 0, 18575.0 / 48384.0, 13525.0 / 55296, 277.0 / 14336, 0.25])

		self._prepare_specific(g00=g00, g01=g01, g11=g11, a=self._a, b=self._b, cval=self._c,
			cerr=self._cs, eps=self._eps, tiny=self._tiny)

	def _cpu__prepare_specific(self, **kwds):
		pass

	def _gpu__prepare_specific(self, **kwds):
		kernel_template = """
			EXPORTED_FUNC void multiplyConstantCS(GLOBAL_MEM COMPLEX *data, SCALAR c)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(val, c);
			}

			EXPORTED_FUNC void multiplyConstantCS_2comp(GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, SCALAR c)
			{
				LIMITED_BY_GRID;
				COMPLEX val;
				val = data0[GLOBAL_INDEX];
				data0[GLOBAL_INDEX] = complex_mul_scalar(val, c);
				val = data1[GLOBAL_INDEX];
				data1[GLOBAL_INDEX] = complex_mul_scalar(val, c);
			}

			EXPORTED_FUNC void transformIP(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				SCALAR e = energy[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(val, exp(e * dt));
			}

			EXPORTED_FUNC void transformIP_2comp(GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY_GRID;
				COMPLEX val;
				SCALAR e = energy[GLOBAL_INDEX];

				val = data0[GLOBAL_INDEX];
				data0[GLOBAL_INDEX] = complex_mul_scalar(val, exp(e * dt));
				val = data1[GLOBAL_INDEX];
				data1[GLOBAL_INDEX] = complex_mul_scalar(val, exp(e * dt));
			}

			EXPORTED_FUNC void calculateScale(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX deriv = k[GLOBAL_INDEX];
				COMPLEX val = data[GLOBAL_INDEX];
				res[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${tiny}
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
					abs(deriv.x) + abs(val.x) + (SCALAR)${tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${tiny}
				);

				deriv = k[GLOBAL_INDEX + ${g.size}];
				val = data1[GLOBAL_INDEX];
				res1[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${tiny}
				);
			}

			EXPORTED_FUNC void calculateError(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *scale)
			{
				LIMITED_BY_GRID;
				COMPLEX val = k[GLOBAL_INDEX];
				COMPLEX s = scale[GLOBAL_INDEX];
				k[GLOBAL_INDEX] = complex_ctr(
					val.x / s.x / (SCALAR)${eps},
					val.y / s.y / (SCALAR)${eps}
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
					val.x / s.x / (SCALAR)${eps},
					val.y / s.y / (SCALAR)${eps}
				);

				val = k[GLOBAL_INDEX + ${g.size}];
				s = scale1[GLOBAL_INDEX];
				k[GLOBAL_INDEX + ${g.size}] = complex_ctr(
					val.x / s.x / (SCALAR)${eps},
					val.y / s.y / (SCALAR)${eps}
				);
			}

			EXPORTED_FUNC void propagationFunc(GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				SCALAR n = squared_abs(val);
				SCALAR p = potentials[GLOBAL_INDEX];
				k[GLOBAL_INDEX + ${g.size} * stage] = complex_mul_scalar(
					val, -dt0 * (p + n * (SCALAR)${g00}));
			}

			EXPORTED_FUNC void propagationFunc_2comp(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *data0, GLOBAL_MEM COMPLEX *data1,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val0 = data0[GLOBAL_INDEX];
				COMPLEX val1 = data1[GLOBAL_INDEX];
				SCALAR n0 = squared_abs(val0);
				SCALAR n1 = squared_abs(val1);
				SCALAR p = potentials[GLOBAL_INDEX];

				k[GLOBAL_INDEX + ${g.size} * stage] = complex_mul_scalar(
					val0, -dt0 * (p + n0 * (SCALAR)${g00} + n1 * (SCALAR)${g01}));
				k[GLOBAL_INDEX + ${g.size} * stage + ${g.size * 6}] = complex_mul_scalar(
					val1, -dt0 * (p + n0 * (SCALAR)${g01} + n1 * (SCALAR)${g11}));
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
						(SCALAR)${b[stage, s]},
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
						(SCALAR)${b[stage, s]},
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
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${cval[s] - cerr[s]});
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
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${cval[s] - cerr[s]});
				%endfor
				res0[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX] = complex_ctr(abs(err_val.x), abs(err_val.y));

				res_val = data1[GLOBAL_INDEX];
				err_val = complex_ctr(0, 0);
				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${g.size * s + g.size * 6}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${cval[s] - cerr[s]});
				%endfor
				res1[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX + ${g.size}] = complex_ctr(abs(err_val.x), abs(err_val.y));
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, **kwds)

		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS
		self._kernel_transformIP = self._program.transformIP
		self._kernel_calculateScale = self._program.calculateScale
		self._kernel_calculateError = self._program.calculateError
		self._kernel_propagationFunc = self._program.propagationFunc
		self._kernel_createData = self._program.createData
		self._kernel_sumResults = self._program.sumResults

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, c):
		data *= c

	def _cpu__kernel_multiplyConstantCS_2comp(self, gsize, data0, data1, c0, c1):
		data0 *= c0
		data1 *= c1

	def _cpu__kernel_transformIP(self, gsize, data, energy, dt):
		data *= numpy.exp(energy * dt)

	def _cpu__kernel_transformIP_2comp(self, gsize, data0, data1, energy, dt):
		coeffs = numpy.exp(energy * dt)
		data0 *= coeffs
		data1 *= coeffs

	def _cpu__kernel_calculateScale(self, gsize, res, k, data):
		res.flat[:] = (
			numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag) +
			numpy.abs(data.real) + 1j * numpy.abs(data.imag) +
			(1 + 1j) * self._tiny).flat

	def _cpu__kernel_calculateScale_2comp(self, gsize, res0, res1, k, data0, data1):
		res0.flat[:] = (numpy.abs(k[0, 0]) + numpy.abs(data0) + tiny).flat
		res1.flat[:] = (numpy.abs(k[1, 0]) + numpy.abs(data1) + tiny).flat

	def _cpu__kernel_calculateError(self, gsize, k, scale):
		shape = k.shape[2:]
		scale = scale.reshape(shape)
		k[0, 0].real /= scale.real * self._eps
		k[0, 0].imag /= scale.imag * self._eps

	def _cpu__kernel_calculateError_2comp(self, gsize, k, scale0, scale1):
		shape = k.shape[2:]
		k[0, 0] /= scale0.reshape(shape) * self._eps
		k[1, 0] /= scale1.reshape(shape) * self._eps

	def _cpu__kernel_propagationFunc(self, gsize, k, data, potentials, dt0, stage):
		g = self._g00
		n = numpy.abs(data) ** 2
		k[0, stage] = -((potentials + n * g) * data) * dt0

	def _cpu__kernel_propagationFunc_2comp(self, gsize, k, data0, data1, potentials, dt0, stage):
		g00 = self._g00
		g01 = self._g01
		g11 = self._g11

		n0 = numpy.abs(data0) ** 2
		n1 = numpy.abs(data1) ** 2

		k[0, stage] = -((potentials + n0 * g00 + n1 * g01) * data0) * dt0
		k[1, stage] = -((potentials + n0 * g01 + n1 * g11) * data1) * dt0

	def _cpu__kernel_createData(self, gsize, res, data, k, stage):
		res.flat[:] = data.flat

		b = self._b[stage, :]
		for s in xrange(stage):
 			res += k[0, s] * b[s]

	def _cpu__kernel_createData_2comp(self, gsize, res0, res1, data0, data1, k, stage):
		res0.flat[:] = data0.flat
		res1.flat[:] = data1.flat

		b = self._b[stage, :]
		for s in xrange(stage):
 			res0 += k[0, s] * b[s]
			res1 += k[1, s] * b[s]

	def _cpu__kernel_sumResults(self, gsize, res, k, data):
		res.flat[:] = data.flat

		c = self._c
		c_err = c - self._cs

		for s in xrange(6):
			res += k[0, s] * c[s]

		k[0, 0] *= c_err[0]
		for s in xrange(1, 6):
			k[0, 0] += k[0, s] * c_err[s]
		k[0, 0] = numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag)

	def _cpu__kernel_sumResults_2comp(self, gsize, res0, res1, k, data0, data1):
		res0.flat[:] = data0.flat
		res1.flat[:] = data1.flat

		c = self._c
		c_err = c - self._cs

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
			dt = self._a[stage] * dt0
			self._fromIP(self._xdata0, None, dt)
			self._kernel_propagationFunc(psi0.size, self._k, self._xdata0,
				self._potentials, cast(dt0), numpy.int32(stage))

		self._kernel_sumResults(psi0.size, self._xdata0,
			self._k, psi0.data)

	def _propagate_rk5_2comp(self, psi0, psi1, dt0):

		cast = self._constants.scalar.cast

		for stage in xrange(6):
			self._kernel_createData_2comp(psi0.size, self._xdata0, self._xdata1,
				psi0.data, psi1.data, self._k, numpy.int32(stage))
			dt = a[stage] * dt0
			self._fromIP(self._xdata0, self._xdata1, dt)
			self._kernel_propagationFunc_2comp(psi0.size, self._k, self._xdata0, self._xdata1,
				self._potentials, cast(dt0), numpy.int32(stage))
			self._toIP(self._xdata0, self._xdata1, dt)

		self._kernel_sumResults_2comp(psi0.size, self._xdata0, self._xdata1,
			self._k, psi0.data, psi1.data)

	def _propagate(self, psi0, psi1):

		safety = 0.9
		eps = self._eps

		dt = self._dt
		cast = self._constants.scalar.cast

		# Estimate scale for this step

		if psi1 is None:
			self._kernel_propagationFunc(psi0.size, self._k, psi0.data, self._potentials,
				cast(dt), numpy.int32(0))
			self._kernel_calculateScale(psi0.size, self._scale0, self._k, psi0.data)
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

		self._dt_used = dt

		if errmax > (5.0 / safety) ** (-1.0 / 0.2):
			self._dt = safety * dt * (errmax ** (-0.2))
		else:
			self._dt = 5.0 * dt

		self._env.copyBuffer(self._xdata0, dest=psi0.data)
		if psi1 is not None:
			self._env.copyBuffer(self._xdata1, dest=psi1.data)

		self._fromIP(psi0.data, psi1.data if psi1 is not None else None, self._dt_used)

	def _toIP(self, data0, data1, dt):
		if dt == 0.0:
			return

		self._plan.execute(data0)
		if data1 is not None:
			self._plan.execute(data1)

		if data1 is not None:
			self._kernel_transformIP_2comp(data0.size, data0, data1,
				self._energy, self._constants.scalar.cast(dt))
		else:
			self._kernel_transformIP(data0.size, data0,
				self._energy, self._constants.scalar.cast(dt))

		self._plan.execute(data0, inverse=True)
		if data1 is not None:
			self._plan.execute(data1, inverse=True)

	def _fromIP(self, data0, data1, dt):
		self._toIP(data0, data1, -dt)

	def _toMeasurementSpace(self, psi0, psi1):
		pass

	def _toEvolutionSpace(self, psi0, psi1):
		pass

	def _renormalize(self, psi0, psi1, c0, c1):
		cast = self._constants.scalar.cast
		if psi1 is None:
			self._kernel_multiplyConstantCS(psi0.size, psi0.data, cast(c0))
		else:
			self._kernel_multiplyConstantCS_2comp(psi0.size, psi0.data, psi1.data, cast(c0), cast(c1))

	def _create(self, psi0, psi1, N0, N1, precision, **kwds):

		two_component = psi1 is not None

		# it would be nice to use two-component TF state here,
		# but the formula is quite complex, and it is much easier
		# just to start from something approximately correct
		self._tf_gs.fillWithTF(psi0, N0)
		if two_component:
			self._tf_gs.fillWithTF(psi1, N1)

		verbose = kwds.pop('verbose', False)
		g_by_hbar = self._constants.g / self._constants.hbar
		kwds['g00'] = g_by_hbar[psi0.comp, psi0.comp]
		if two_component:
			kwds['g01'] = g_by_hbar[psi0.comp, psi1.comp]
			kwds['g11'] = g_by_hbar[psi1.comp, psi1.comp]

		# Criterion for 'tiny' limit is a bit different for imaginary time method
		# We want propagation to be accurate, but if the steps are too small,
		# the precision will be reached too soon.
		# So we are setting quite a big 'tiny' and hoping that it will be ok.
		peak = numpy.abs(self._env.fromDevice(psi0.data)).max()
		if two_component:
			peak = min(peak, numpy.abs(self._env.fromDevice(psi1.data)).max())

		self._tiny = peak / 1e0

		self._prepare(**kwds)

		stats = self._statistics

		if two_component:
			total_N = lambda psi0, psi1: stats.getN(psi0) + stats.getN(psi1)
			total_E = lambda psi0, psi1, N: stats.getEnergy2comp(psi0, psi1, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu2comp(psi0, psi1, N=N)
		else:
			total_N = lambda psi0, psi1: stats.getN(psi0)
			total_E = lambda psi0, psi1, N: stats.getEnergy(psi0, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu(psi0, N=N)

		E = 0.0
		new_E = total_E(psi0, psi1, N0 + N1)
		self._toEvolutionSpace(psi0, psi1)

		# Reducing the dependence on time step
		# Now we can use small time steps not being afraid that
		# propagation will be terminated too soon (because dE is too small)
		# (TODO: dE ~ dt, but not exactly; see W. Bao and Q. Du, 2004, eqn. 2.7
		# Now default precision is chosen so that usual dt's work well with it)
		while abs(E - new_E) / new_E > precision * self._dt:

			self._propagate(psi0, psi1)
			self._toMeasurementSpace(psi0, psi1)

			# renormalize
			if two_component:
				new_N0 = stats.getN(psi0)
				new_N1 = stats.getN(psi1)
				c0 = numpy.sqrt(N0 / new_N0)
				c1 = numpy.sqrt(N1 / new_N1)
				self._renormalize(psi0, psi1, c0, c1)
			else:
				new_N0 = stats.getN(psi0)
				self._renormalize(psi0, psi1, numpy.sqrt(N0 / new_N0), None)

			E = new_E
			new_E = total_E(psi0, psi1, N0 + N1)
			self._toEvolutionSpace(psi0, psi1)
			if new_E > E:
				print "Warning: energy starts to rise, propagation aborted"
				break

		self._toMeasurementSpace(psi0, psi1)

		if verbose:
			postfix = "(two components)" if two_component else "(one component)"
			pop = str(stats.getN(psi0)) + " + " + str(stats.getN(psi1)) if two_component else \
				str(stats.getN(psi0))

			print "Ground state calculation " + postfix + " :" + \
					" N = " + N + \
					" E = " + str(total_E(psi0, psi1, N0 + N1)) + \
					" mu = " + str(total_mu(psi0, psi1, N0 + N1))

	def create(self, N, comp=0, precision=1e-1, **kwds):
		psi = Wavefunction(self._env, self._constants, self._grid, comp=comp)
		self._create(psi, None, N, 0, precision, **kwds)
		return psi

	def createCloud(self, N, ratio=1.0, precision=1e-1, **kwds):
		cloud = TwoComponentCloud(self._env, self._constants, self._grid)
		if ratio == 1.0:
			self._create(cloud.psi0, None, N, 0, precision, **kwds)
		else:
			self._create(cloud.psi0, cloud.psi1, N * ratio, N * (1.0 - ratio),
				precision, **kwds)
		return cloud


class RK5HarmonicGroundState(PairedCalculation):

	def __init__(self, env, constants, grid, dt_guess=1e-4, eps=1e-9):
		PairedCalculation.__init__(self, env)

		assert isinstance(grid, HarmonicGrid)

		self._constants = constants.copy()
		self._grid = grid.copy()

		self._dt = dt_guess
		self._eps = eps

		self._tf_gs = TFGroundState(env, constants, grid)
		self._statistics = ParticleStatistics(env, constants, grid)

		self._energy = getHarmonicEnergy(self._env, self._constants, self._grid)
		self._maxFinder = createMaxFinder(self._env, self._constants.scalar.dtype)

		shape = self._grid.mshape
		cdtype = self._constants.complex.dtype
		sdtype = self._constants.scalar.dtype

		self._xdata0 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._xdata1 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._x3data0 = self._env.allocate((1,) + grid.shapes[3], dtype=cdtype)

		self._k = self._env.allocate((2, 6) + shape, dtype=cdtype)

		self._scale0 = self._env.allocate((1,) + shape, dtype=cdtype)
		self._scale1 = self._env.allocate((1,) + shape, dtype=cdtype)

		self._plan3 = createFHTPlan(env, constants, grid, 3)

	def _prepare(self, g00=0.0, g01=0.0, g11=0.0):
		self._dt_used = 0

		self._g00 = g00
		self._g01 = g01
		self._g11 = g11

		self._a = numpy.array([0, 0.2, 0.3, 0.6, 1, 0.875])
		self._b = numpy.array([
			[0, 0, 0, 0, 0],
			[1.0 / 5, 0, 0, 0, 0],
			[3.0 / 40, 9.0 / 40, 0, 0, 0],
			[3.0 / 10, -9.0 / 10, 6.0 / 5, 0, 0],
			[-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27, 0],
			[1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]
		])
		self._c = numpy.array([37.0 / 378, 0, 250.0 / 621, 125.0 / 594, 0, 512.0 / 1771])
		self._cs = numpy.array([2825.0 / 27648, 0, 18575.0 / 48384.0, 13525.0 / 55296, 277.0 / 14336, 0.25])

		self._prepare_specific(g00=g00, g01=g01, g11=g11, a=self._a, b=self._b, cval=self._c,
			cerr=self._cs, eps=self._eps, tiny=self._tiny)

	def _cpu__prepare_specific(self, **kwds):
		pass

	def _gpu__prepare_specific(self, **kwds):
		kernel_template = """
			EXPORTED_FUNC void multiplyConstantCS(GLOBAL_MEM COMPLEX *data, SCALAR c)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(val, c);
			}

			EXPORTED_FUNC void multiplyConstantCS_2comp(GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, SCALAR c)
			{
				LIMITED_BY_GRID;
				COMPLEX val;
				val = data0[GLOBAL_INDEX];
				data0[GLOBAL_INDEX] = complex_mul_scalar(val, c);
				val = data1[GLOBAL_INDEX];
				data1[GLOBAL_INDEX] = complex_mul_scalar(val, c);
			}

			EXPORTED_FUNC void transformIP(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				SCALAR e = energy[GLOBAL_INDEX];
				data[GLOBAL_INDEX] = complex_mul_scalar(val, exp(e * dt));
			}

			EXPORTED_FUNC void transformIP_2comp(GLOBAL_MEM COMPLEX *data0,
				GLOBAL_MEM COMPLEX *data1, GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY_GRID;
				COMPLEX val;
				SCALAR e = energy[GLOBAL_INDEX];

				val = data0[GLOBAL_INDEX];
				data0[GLOBAL_INDEX] = complex_mul_scalar(val, exp(e * dt));
				val = data1[GLOBAL_INDEX];
				data1[GLOBAL_INDEX] = complex_mul_scalar(val, exp(e * dt));
			}

			EXPORTED_FUNC void calculateScale(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX deriv = k[GLOBAL_INDEX];
				COMPLEX val = data[GLOBAL_INDEX];
				res[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${tiny}
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
					abs(deriv.x) + abs(val.x) + (SCALAR)${tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${tiny}
				);

				deriv = k[GLOBAL_INDEX + ${g.size}];
				val = data1[GLOBAL_INDEX];
				res1[GLOBAL_INDEX] = complex_ctr(
					abs(deriv.x) + abs(val.x) + (SCALAR)${tiny},
					abs(deriv.y) + abs(val.y) + (SCALAR)${tiny}
				);
			}

			EXPORTED_FUNC void calculateError(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *scale)
			{
				LIMITED_BY_GRID;
				COMPLEX val = k[GLOBAL_INDEX];
				COMPLEX s = scale[GLOBAL_INDEX];
				k[GLOBAL_INDEX] = complex_ctr(
					val.x / s.x / (SCALAR)${eps},
					val.y / s.y / (SCALAR)${eps}
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
					val.x / s.x / (SCALAR)${eps},
					val.y / s.y / (SCALAR)${eps}
				);

				val = k[GLOBAL_INDEX + ${g.size}];
				s = scale1[GLOBAL_INDEX];
				k[GLOBAL_INDEX + ${g.size}] = complex_ctr(
					val.x / s.x / (SCALAR)${eps},
					val.y / s.y / (SCALAR)${eps}
				);
			}

			EXPORTED_FUNC void calculateNonlinear(GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY_GRID;
				COMPLEX val = data[GLOBAL_INDEX];
				SCALAR n = squared_abs(val);
				data[GLOBAL_INDEX] = complex_mul_scalar(val, -n * (SCALAR)${g00});
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

			EXPORTED_FUNC void propagationFunc_2comp(GLOBAL_MEM COMPLEX *k,
				GLOBAL_MEM COMPLEX *data0, GLOBAL_MEM COMPLEX *data1,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0, int stage)
			{
				LIMITED_BY_GRID;
				COMPLEX val0 = data0[GLOBAL_INDEX];
				COMPLEX val1 = data1[GLOBAL_INDEX];
				SCALAR n0 = squared_abs(val0);
				SCALAR n1 = squared_abs(val1);
				SCALAR p = potentials[GLOBAL_INDEX];

				k[GLOBAL_INDEX + ${g.size} * stage] = complex_mul_scalar(
					val0, -dt0 * (p + n0 * (SCALAR)${g00} + n1 * (SCALAR)${g01}));
				k[GLOBAL_INDEX + ${g.size} * stage + ${g.size * 6}] = complex_mul_scalar(
					val1, -dt0 * (p + n0 * (SCALAR)${g01} + n1 * (SCALAR)${g11}));
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
						(SCALAR)${b[stage, s]},
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
						(SCALAR)${b[stage, s]},
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
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${cval[s] - cerr[s]});
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
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${cval[s] - cerr[s]});
				%endfor
				res0[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX] = complex_ctr(abs(err_val.x), abs(err_val.y));

				res_val = data1[GLOBAL_INDEX];
				err_val = complex_ctr(0, 0);
				%for s in xrange(6):
					kval = k[GLOBAL_INDEX + ${g.size * s + g.size * 6}];
					res_val = res_val + complex_mul_scalar(kval, (SCALAR)${cval[s]});
					err_val = err_val + complex_mul_scalar(kval, (SCALAR)${cval[s] - cerr[s]});
				%endfor
				res1[GLOBAL_INDEX] = res_val;
				k[GLOBAL_INDEX + ${g.size}] = complex_ctr(abs(err_val.x), abs(err_val.y));
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, **kwds)

		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS
		self._kernel_transformIP = self._program.transformIP
		self._kernel_calculateScale = self._program.calculateScale
		self._kernel_calculateError = self._program.calculateError
		self._kernel_propagationFunc = self._program.propagationFunc
		self._kernel_calculateNonlinear = self._program.calculateNonlinear
		self._kernel_createData = self._program.createData
		self._kernel_sumResults = self._program.sumResults

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, c):
		data *= c

	def _cpu__kernel_multiplyConstantCS_2comp(self, gsize, data0, data1, c0, c1):
		data0 *= c0
		data1 *= c1

	def _cpu__kernel_transformIP(self, gsize, data, energy, dt):
		data *= numpy.exp(energy * dt)

	def _cpu__kernel_transformIP_2comp(self, gsize, data0, data1, energy, dt):
		coeffs = numpy.exp(energy * dt)
		data0 *= coeffs
		data1 *= coeffs

	def _cpu__kernel_calculateScale(self, gsize, res, k, data):
		res.flat[:] = (
			numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag) +
			numpy.abs(data.real) + 1j * numpy.abs(data.imag) +
			(1 + 1j) * self._tiny).flat

	def _cpu__kernel_calculateScale_2comp(self, gsize, res0, res1, k, data0, data1):
		res0.flat[:] = (numpy.abs(k[0, 0]) + numpy.abs(data0) + tiny).flat
		res1.flat[:] = (numpy.abs(k[1, 0]) + numpy.abs(data1) + tiny).flat

	def _cpu__kernel_calculateError(self, gsize, k, scale):
		shape = k.shape[2:]
		scale = scale.reshape(shape)
		k[0, 0].real /= scale.real * self._eps
		k[0, 0].imag /= scale.imag * self._eps

	def _cpu__kernel_calculateError_2comp(self, gsize, k, scale0, scale1):
		shape = k.shape[2:]
		k[0, 0] /= scale0.reshape(shape) * self._eps
		k[1, 0] /= scale1.reshape(shape) * self._eps

	def _cpu__kernel_calculateNonlinear(self, gsize, data):
		g = self._g00
		n = numpy.abs(data) ** 2
		data *= -(n * g)

	def _cpu__kernel_propagationFunc(self, gsize, k, data, nldata, energy, dt0, stage):
		k[0, stage] = (-data * energy + nldata) * dt0

	def _cpu__kernel_propagationFunc_2comp(self, gsize, k, data0, data1, potentials, dt0, stage):
		g00 = self._g00
		g01 = self._g01
		g11 = self._g11

		n0 = numpy.abs(data0) ** 2
		n1 = numpy.abs(data1) ** 2

		k[0, stage] = -((potentials + n0 * g00 + n1 * g01) * data0) * dt0
		k[1, stage] = -((potentials + n0 * g01 + n1 * g11) * data1) * dt0

	def _cpu__kernel_createData(self, gsize, res, data, k, stage):
		res.flat[:] = data.flat

		b = self._b[stage, :]
		for s in xrange(stage):
 			res += k[0, s] * b[s]

	def _cpu__kernel_createData_2comp(self, gsize, res0, res1, data0, data1, k, stage):
		res0.flat[:] = data0.flat
		res1.flat[:] = data1.flat

		b = self._b[stage, :]
		for s in xrange(stage):
 			res0 += k[0, s] * b[s]
			res1 += k[1, s] * b[s]

	def _cpu__kernel_sumResults(self, gsize, res, k, data):
		res.flat[:] = data.flat

		c = self._c
		c_err = c - self._cs

		for s in xrange(6):
			res += k[0, s] * c[s]

		k[0, 0] *= c_err[0]
		for s in xrange(1, 6):
			k[0, 0] += k[0, s] * c_err[s]
		k[0, 0] = numpy.abs(k[0, 0].real) + 1j * numpy.abs(k[0, 0].imag)

	def _cpu__kernel_sumResults_2comp(self, gsize, res0, res1, k, data0, data1):
		res0.flat[:] = data0.flat
		res1.flat[:] = data1.flat

		c = self._c
		c_err = c - self._cs

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
			dt = self._a[stage] * dt0
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
			dt = a[stage] * dt0
			self._fromIP(self._xdata0, self._xdata1, dt)
			self._kernel_propagationFunc_2comp(psi0.size, self._k, self._xdata0, self._xdata1,
				self._potentials, cast(dt0), numpy.int32(stage))
			self._toIP(self._xdata0, self._xdata1, dt)

		self._kernel_sumResults_2comp(psi0.size, self._xdata0, self._xdata1,
			self._k, psi0.data, psi1.data)

	def _propagate(self, psi0, psi1):

		safety = 0.9
		eps = self._eps

		dt = self._dt
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

		self._dt_used = dt

		if errmax > (5.0 / safety) ** (-1.0 / 0.2):
			self._dt = safety * dt * (errmax ** (-0.2))
		else:
			self._dt = 5.0 * dt

		self._env.copyBuffer(self._xdata0, dest=psi0.data)
		if psi1 is not None:
			self._env.copyBuffer(self._xdata1, dest=psi1.data)

	def _toMeasurementSpace(self, psi0, psi1):
		psi0.toXSpace()
		if psi1 is not None:
			psi1.toXSpace()

	def _toEvolutionSpace(self, psi0, psi1):
		psi0.toMSpace()
		if psi1 is not None:
			psi1.toMSpace()

	def _renormalize(self, psi0, psi1, c0, c1):
		cast = self._constants.scalar.cast
		if psi1 is None:
			self._kernel_multiplyConstantCS(psi0.size, psi0.data, cast(c0))
		else:
			self._kernel_multiplyConstantCS_2comp(psi0.size, psi0.data, psi1.data, cast(c0), cast(c1))

	def _create(self, psi0, psi1, N0, N1, precision, **kwds):

		two_component = psi1 is not None

		# it would be nice to use two-component TF state here,
		# but the formula is quite complex, and it is much easier
		# just to start from something approximately correct
		self._tf_gs.fillWithTF(psi0, N0)
		if two_component:
			self._tf_gs.fillWithTF(psi1, N1)
		self._toEvolutionSpace(psi0, psi1)

		verbose = kwds.pop('verbose', False)
		g_by_hbar = self._constants.g / self._constants.hbar
		kwds['g00'] = g_by_hbar[psi0.comp, psi0.comp]
		if two_component:
			kwds['g01'] = g_by_hbar[psi0.comp, psi1.comp]
			kwds['g11'] = g_by_hbar[psi1.comp, psi1.comp]

		# Criterion for 'tiny' limit is a bit different for imaginary time method
		# We want propagation to be accurate, but if the steps are too small,
		# the precision will be reached too soon.
		# So we are setting quite a big 'tiny' and hoping that it will be ok.
		peak = numpy.abs(self._env.fromDevice(psi0.data)).max()
		if two_component:
			peak = min(peak, numpy.abs(self._env.fromDevice(psi1.data)).max())

		self._tiny = peak / 1e4

		self._prepare(**kwds)

		stats = self._statistics

		if two_component:
			total_N = lambda psi0, psi1: stats.getN(psi0) + stats.getN(psi1)
			total_E = lambda psi0, psi1, N: stats.getEnergy2comp(psi0, psi1, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu2comp(psi0, psi1, N=N)
		else:
			total_N = lambda psi0, psi1: stats.getN(psi0)
			total_E = lambda psi0, psi1, N: stats.getEnergy(psi0, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu(psi0, N=N)

		self._toMeasurementSpace(psi0, psi1)
		E = 0.0
		new_E = total_E(psi0, psi1, N0 + N1)
		self._toEvolutionSpace(psi0, psi1)

		# Reducing the dependence on time step
		# Now we can use small time steps not being afraid that
		# propagation will be terminated too soon (because dE is too small)
		# (TODO: dE ~ dt, but not exactly; see W. Bao and Q. Du, 2004, eqn. 2.7
		# Now default precision is chosen so that usual dt's work well with it)
		while abs(E - new_E) / new_E > precision * self._dt:

			self._propagate(psi0, psi1)
			self._toMeasurementSpace(psi0, psi1)

			# renormalize
			if two_component:
				new_N0 = stats.getN(psi0)
				new_N1 = stats.getN(psi1)
				c0 = numpy.sqrt(N0 / new_N0)
				c1 = numpy.sqrt(N1 / new_N1)
				self._renormalize(psi0, psi1, c0, c1)
			else:
				new_N0 = stats.getN(psi0)
				self._renormalize(psi0, psi1, numpy.sqrt(N0 / new_N0), None)

			E = new_E
			new_E = total_E(psi0, psi1, N0 + N1)
#			print new_E
			self._toEvolutionSpace(psi0, psi1)
			if new_E > E:
				print "Warning: energy starts to rise, propagation aborted"
				break

		self._toMeasurementSpace(psi0, psi1)

		if verbose:
			postfix = "(two components)" if two_component else "(one component)"
			pop = str(stats.getN(psi0)) + " + " + str(stats.getN(psi1)) if two_component else \
				str(stats.getN(psi0))

			print "Ground state calculation " + postfix + " :" + \
					" N = " + N + \
					" E = " + str(total_E(psi0, psi1, N0 + N1)) + \
					" mu = " + str(total_mu(psi0, psi1, N0 + N1))

	def create(self, N, comp=0, precision=1e-1, **kwds):
		psi = Wavefunction(self._env, self._constants, self._grid, comp=comp)
		self._create(psi, None, N, 0, precision, **kwds)
		return psi

	def createCloud(self, N, ratio=1.0, precision=1e-1, **kwds):
		cloud = TwoComponentCloud(self._env, self._constants, self._grid)
		if ratio == 1.0:
			self._create(cloud.psi0, None, N, 0, precision, **kwds)
		else:
			self._create(cloud.psi0, cloud.psi1, N * ratio, N * (1.0 - ratio),
				precision, **kwds)
		return cloud