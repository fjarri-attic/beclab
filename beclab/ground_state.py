"""
Ground state calculation classes
"""

import copy
import numpy

from .helpers import *
from .wavefunction import Wavefunction, TwoComponentCloud
from .meters import ParticleStatistics
from .constants import getPotentials, getPlaneWaveEnergy, UniformGrid, HarmonicGrid


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

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
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


class SplitStepGroundState(PairedCalculation):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)

		assert isinstance(grid, UniformGrid)

		self._constants = constants.copy()
		self._grid = grid.copy()

		self._tf_gs = TFGroundState(env, constants, grid)
		self._statistics = ParticleStatistics(env, constants, grid)

	def _cpu__prepare_specific(self, **kwds):
		self._dt = kwds['dt']
		self._g00 = kwds['g00']
		self._g01 = kwds['g01']
		self._g11 = kwds['g11']
		self._itmax = kwds['itmax']

	def _gpu__prepare_specific(self, **kwds):
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
				%for i in range(itmax):
					// calculate midpoint log derivative and exponentiate
					dval = exp((SCALAR)${dt / 2.0} *
						(-V - (SCALAR)${g00} * squared_abs(val)));

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
				%for i in range(itmax):
					// calculate midpoint log derivative and exponentiate
					n0 = squared_abs(val0);
					n1 = squared_abs(val1);

					dval0 = exp((SCALAR)${dt / 2.0} *
						(-V - (SCALAR)${g00} * n0 - (SCALAR)${g01} * n1));
					dval1 = exp((SCALAR)${dt / 2.0} *
						(-V - (SCALAR)${g01} * n0 - (SCALAR)${g11} * n1));

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
			self._grid, dt=kwds['dt'], g00=kwds['g00'], g01=kwds['g01'], g11=kwds['g11'],
			itmax=kwds['itmax'])

		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS
		self._kernel_multiplyConstantCS_2comp = self._program.multiplyConstantCS_2comp
		self._kernel_mpropagate = self._program.mpropagate
		self._kernel_mpropagate_2comp = self._program.mpropagate_2comp
		self._kernel_xpropagate = self._program.xpropagate
		self._kernel_xpropagate_2comp = self._program.xpropagate_2comp

	def _prepare(self, dt=1e-5, g00=0.0, g01=0.0, g11=0.0, itmax=3):
		self._potentials = getPotentials(self._env, self._constants, self._grid)
		energy = getPlaneWaveEnergy(None, self._constants, self._grid)
		self._mode_prop = self._env.toDevice(numpy.exp(energy * (-dt / 2)))
		self._prepare_specific(dt=dt, g00=g00, g01=g01, g11=g11, itmax=itmax)

	def _cpu__kernel_multiplyConstantCS(self, gsize, data, c):
		data *= c

	def _cpu__kernel_multiplyConstantCS_2comp(self, gsize, data0, data1, c0, c1):
		data0 *= c0
		data1 *= c1

	def _cpu__kernel_mpropagate(self, gsize, data0, mode_prop):
		data0 *= mode_prop

	def _cpu__kernel_mpropagate_2comp(self, gsize, data0, data1, mode_prop):
		data0 *= mode_prop
		data1 *= mode_prop

	def _cpu__kernel_xpropagate(self, gsize, data, potentials):
		data_copy = data.copy()
		g = self._g00
		dt = -self._dt / 2

		for i in xrange(self._itmax):
			n = numpy.abs(data) ** 2
			d = numpy.exp((potentials + n * g) * dt)
			data.flat[:] = (data_copy * d).flat
		data *= d

	def _cpu__kernel_xpropagate_2comp(self, gsize, data0, data1, potentials):

		dt = -self._dt / 2
		g00 = self._g00
		g01 = self._g01
		g11 = self._g11

		data0_copy = data0.copy()
		data1_copy = data1.copy()

		for i in xrange(self._constants.itmax):
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

	def _renormalize(self, psi0, psi1, c0, c1):
		cast = self._constants.scalar.cast
		if psi1 is None:
			self._kernel_multiplyConstantCS(psi0.size, psi0.data, cast(c0))
		else:
			self._kernel_multiplyConstantCS_2comp(psi0.size, psi0.data, psi1.data, cast(c0), cast(c1))

	def _create(self, psi0, psi1, N0, N1, precision, **kwds):

		two_component = psi1 is not None
		verbose = kwds.pop('verbose', False)

		g_by_hbar = self._constants.g / self._constants.hbar
		kwds['g00'] = g_by_hbar[psi0.comp, psi0.comp]
		if two_component:
			kwds['g01'] = g_by_hbar[psi0.comp, psi1.comp]
			kwds['g11'] = g_by_hbar[psi1.comp, psi1.comp]
		self._prepare(**kwds)

		# it would be nice to use two-component TF state here,
		# but the formula is quite complex, and it is much easier
		# just to start from something approximately correct
		self._tf_gs.fillWithTF(psi0, N0)
		if two_component:
			self._tf_gs.fillWithTF(psi1, N1)

		stats = self._statistics

		if two_component:
			total_N = lambda psi0, psi1: stats.getN(psi0) + stats.getN(psi1)
			total_E = lambda psi0, psi1, N: stats.getEnergy2comp(psi0, psi1, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu2comp(psi0, psi1, N=N)
			to_xspace = lambda psi0, psi1: psi0.toXSpace(), psi1.toXSpace()
			to_mspace = lambda psi0, psi1: psi0.toMSpace(), psi1.toMSpace()
		else:
			total_N = lambda psi0, psi1: stats.getN(psi0)
			total_E = lambda psi0, psi1, N: stats.getEnergy(psi0, N=N)
			total_mu = lambda psi0, psi1, N: stats.getMu(psi0, N=N)
			to_xspace = lambda psi0, psi1: psi0.toXSpace()
			to_mspace = lambda psi0, psi1: psi0.toMSpace()

		E = 0.0
		new_E = total_E(psi0, psi1, N0 + N1)

		to_mspace(psi0, psi1)
		while abs(E - new_E) / new_E > precision:

			# propagation
			self._mpropagate(psi0, psi1)
			to_xspace(psi0, psi1)
			self._xpropagate(psi0, psi1)
			to_mspace(psi0, psi1)
			self._mpropagate(psi0, psi1)

			# normalization
			to_xspace(psi0, psi1)

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
			to_mspace(psi0, psi1)

		to_xspace(psi0, psi1)

		if verbose:
			postfix = "(two components)" if two_component else "(one component)"
			pop = str(stats.getN(psi0)) + " + " + str(stats.getN(psi1)) if two_component else \
				str(stats.getN(psi0))

			print "Ground state calculation " + postfix + " :" + \
					" N = " + N + \
					" E = " + str(total_E(psi0, psi1, N0 + N1)) + \
					" mu = " + str(total_mu(psi0, psi1, N0 + N1))

	def create(self, N, comp=0, precision=1e-6, dt=1e-5):
		psi = Wavefunction(self._env, self._constants, self._grid, comp=comp)
		self._create(psi, None, N, 0, precision, dt=dt)
		return psi

	def createCloud(self, N, ratio=1.0, precision=1e-6, dt=1e-5):
		cloud = TwoComponentCloud(self._env, self._constants, self._grid)
		if ratio == 1.0:
			self._create(cloud.psi0, None, N, 0, precision, dt=dt)
		else:
			self._create(cloud.psi0, cloud.psi1, N * ratio, N * (1.0 - ratio),
				precision, dt=dt)
		return cloud
