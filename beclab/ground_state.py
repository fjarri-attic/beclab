"""
Ground state calculation classes
"""

import copy
import numpy

from .helpers import *
from .wavefunction import Wavefunction, TwoComponentCloud
from .meters import ParticleStatistics
from .constants import getPotentials, UniformGrid, HarmonicGrid


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
			EXPORTED_FUNC void fillWithTFGroundState(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR mu_by_hbar,
				SCALAR g_by_hbar)
			{
				DEFINE_INDEXES;

				SCALAR potential = potentials[cell_index];

				SCALAR e = mu_by_hbar - potential;
				if(e > 0)
					data[index] = complex_ctr(sqrt(e / g_by_hbar), 0);
				else
					data[index] = complex_ctr(0, 0);
			}

			EXPORTED_FUNC void multiplyByScalar(GLOBAL_MEM COMPLEX *data, SCALAR coeff)
			{
				DEFINE_INDEXES;
				COMPLEX x = data[index];
				data[index] = complex_mul_scalar(x, coeff);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants)
		self._kernel_fillWithTFGroundState = self._program.fillWithTFGroundState
		self._kernel_multiplyByScalar = self._program.multiplyByScalar

	def _cpu__kernel_fillWithTFGroundState(self, _, data, potentials, mu_by_hbar, g_by_hbar):
		mask_func = lambda x: 0.0 if x < 0 else x
		mask_map = numpy.vectorize(mask_func)
		self._env.copyBuffer(
			numpy.sqrt(mask_map(mu_by_hbar - self._potentials) / g_by_hbar),
			dest=data)

	def _cpu__kernel_multiplyByScalar(self, _, data, coeff):
		data *= coeff

	def _create(self, data, g, mu):
		cast = self._constants.scalar.cast
		mu_by_hbar = cast(mu / self._constants.hbar)
		g_by_hbar = cast(g / self._constants.hbar)

		self._kernel_fillWithTFGroundState(data.size, data,
			self._potentials, mu_by_hbar, g_by_hbar)

	def create(self, N, comp=0):
		res = Wavefunction(self._env, self._constants, self._grid, comp=comp)
		g = self._constants.g[comp, comp]
		mu = self._constants.muTF(N, dim=self._grid.dim, comp=comp)
		self._create(res.data, g, mu)

		# The total number of atoms is equal to the number requested
		# only in the limit of infinite number of lattice points.
		# So we have to renormalize the data, and we will do it in mode space
		# because lattice spacing is uniform there
		# (an because it is a primary space for harmonic grid).
		res.toMSpace()
		N_real = self._stats.getN(res)
		coeff = numpy.sqrt(N / N_real)
		self._kernel_multiplyByScalar(res.size, res.data, self._constants.scalar.cast(coeff))
		res.toXSpace()

		return res


class GPEGroundState(PairedCalculation):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._tf_gs = TFGroundState(env, constants)
		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
		self._statistics = ParticleStatistics(env, constants)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		self._k_coeff = numpy.exp(self._kvectors * (-self._constants.dt_steady / 2))

	def _gpu__prepare(self):
		kernel_template = """
			EXPORTED_FUNC void multiply(GLOBAL_MEM COMPLEX *data, SCALAR coeff)
			{
				DEFINE_INDEXES;
				data[index] = complex_mul_scalar(data[index], coeff);
			}

			EXPORTED_FUNC void multiply2(GLOBAL_MEM COMPLEX *data1, GLOBAL_MEM COMPLEX *data2,
				SCALAR c1, SCALAR c2)
			{
				DEFINE_INDEXES;
				data1[index] = complex_mul_scalar(data1[index], c1);
				data2[index] = complex_mul_scalar(data2[index], c2);
			}

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			EXPORTED_FUNC void propagateKSpace(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *kvectors)
			{
				DEFINE_INDEXES;

				SCALAR kvector = kvectors[cell_index];

				SCALAR prop_coeff = exp(kvector *
					(SCALAR)${-c.dt_steady / 2.0});
				COMPLEX temp = data[index];
				data[index] = complex_mul_scalar(temp, prop_coeff);
			}

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			// Version for processing two components at once
			EXPORTED_FUNC void propagateKSpace2(
				GLOBAL_MEM COMPLEX *data1, GLOBAL_MEM COMPLEX *data2,
				GLOBAL_MEM SCALAR *kvectors)
			{
				DEFINE_INDEXES;

				SCALAR kvector = kvectors[cell_index];

				SCALAR prop_coeff = exp(kvector *
					(SCALAR)${-c.dt_steady / 2.0});

				data1[index] = complex_mul_scalar(data1[index], prop_coeff);
				data2[index] = complex_mul_scalar(data2[index], prop_coeff);
			}

			// Propagates state in x-space for steady state calculation
			EXPORTED_FUNC void propagateXSpace(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR g_by_hbar)
			{
				DEFINE_INDEXES;

				COMPLEX a = data[index];

				//store initial x-space field
				COMPLEX a0 = a;

				SCALAR da;
				SCALAR V = potentials[cell_index];

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					da = exp((SCALAR)${c.dt_steady / 2.0} *
						(-V - g_by_hbar * squared_abs(a)));

					//propagate to midpoint using log derivative
					a = complex_mul_scalar(a0, da);
				%endfor

				//propagate to endpoint using log derivative
				data[index] = complex_mul_scalar(a, da);
			}

			// Propagates state in x-space for steady state calculation
			EXPORTED_FUNC void propagateXSpace2(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b, GLOBAL_MEM SCALAR *potentials,
				SCALAR g11_by_hbar, SCALAR g22_by_hbar,
				SCALAR g12_by_hbar)
			{
				DEFINE_INDEXES;

				COMPLEX a_res = a[index];
				COMPLEX b_res = b[index];

				//store initial x-space field
				COMPLEX a0 = a_res;
				COMPLEX b0 = b_res;

				SCALAR da, db, a_density, b_density;
				SCALAR V = potentials[cell_index];

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					a_density = squared_abs(a_res);
					b_density = squared_abs(b_res);

					da = exp((SCALAR)${c.dt_steady / 2.0} *
						(-V - g11_by_hbar * a_density - g12_by_hbar * b_density));
					db = exp((SCALAR)${c.dt_steady / 2.0} *
						(-V - g12_by_hbar * a_density - g22_by_hbar * b_density));

					//propagate to midpoint using log derivative
					a_res = complex_mul_scalar(a0, da);
					b_res = complex_mul_scalar(b0, db);
				%endfor

				//propagate to endpoint using log derivative
				a[index] = complex_mul_scalar(a_res, da);
				b[index] = complex_mul_scalar(b_res, db);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants)

		self._propagateKSpace = self._program.propagateKSpace
		self._propagateKSpace2 = self._program.propagateKSpace2
		self._propagateXSpace = self._program.propagateXSpace
		self._propagateXSpace2 = self._program.propagateXSpace2
		self._multiply = self._program.multiply
		self._multiply2 = self._program.multiply2

	def _cpu__kpropagate(self, state1, state2):
		# for numpy arrays, '*=' operator is inplace
		state1.data *= self._k_coeff
		if state2 is not None:
			state2.data *= self._k_coeff

	def _gpu__kpropagate(self, state1, state2):
		if state2 is None:
			self._propagateKSpace(state1.size, state1.data, self._kvectors)
		else:
			self._propagateKSpace2(state1.size, state1.data, state2.data, self._kvectors)

	def _cpu__xpropagate(self, state1, state2):
		p = self._potentials
		dt = -self._constants.dt_steady / 2

		if state2 is None:
			a0 = state1.data.copy()
			g_by_hbar = self._constants.g_by_hbar[(state1.comp, state1.comp)]

			for iter in xrange(self._constants.itmax):
				n = numpy.abs(state1.data) ** 2
				da = numpy.exp((p + n * g_by_hbar) * dt)
				state1.data = a0 * da
			state1.data *= da
		else:
			a0 = state1.data.copy()
			b0 = state2.data.copy()

			comp1 = state1.comp
			comp2 = state2.comp
			g_by_hbar = self._constants.g_by_hbar
			g11_by_hbar = g_by_hbar[(comp1, comp1)]
			g12_by_hbar = g_by_hbar[(comp1, comp2)]
			g22_by_hbar = g_by_hbar[(comp2, comp2)]

			for iter in xrange(self._constants.itmax):
				na = numpy.abs(state1.data) ** 2
				nb = numpy.abs(state2.data) ** 2

				pa = p + na * g11_by_hbar + nb * g12_by_hbar
				pb = p + nb * g22_by_hbar + na * g12_by_hbar

				da = numpy.exp(pa * dt)
				db = numpy.exp(pb * dt)

				state1.data = a0 * da
				state2.data = b0 * db

			state1.data *= da
			state2.data *= db

	def _gpu__xpropagate(self, state1, state2):
		cast = self._constants.scalar.cast
		if state2 is None:
			g_by_hbar = self._constants.g_by_hbar[(state1.comp, state1.comp)]
			self._propagateXSpace(state1.size, state1.data, self._potentials,
				cast(g_by_hbar))
		else:
			comp1 = state1.comp
			comp2 = state2.comp
			g_by_hbar = self._constants.g_by_hbar
			g11_by_hbar = g_by_hbar[(comp1, comp1)]
			g12_by_hbar = g_by_hbar[(comp1, comp2)]
			g22_by_hbar = g_by_hbar[(comp2, comp2)]

			self._propagateXSpace2(state1.size, state1.data, state2.data,
				self._potentials, cast(g11_by_hbar), cast(g22_by_hbar), cast(g12_by_hbar))

	def _cpu__renormalize(self, state1, state2, coeff):
		if state2 is None:
			state1.data *= coeff
		else:
			c1, c2 = coeff
			state1.data *= c1
			state2.data *= c2

	def _gpu__renormalize(self, state1, state2, coeff):
		cast = self._constants.scalar.cast
		if state2 is None:
			self._multiply(state1.size, state1.data, cast(coeff))
		else:
			c1, c2 = coeff
			self._multiply2(state1.size, state1.data, state2.data, cast(c1), cast(c2))

	def _toXSpace(self, state1, state2):
		self._plan.execute(state1.data)
		if state2 is not None:
			self._plan.execute(state2.data)

	def _toKSpace(self, state1, state2):
		self._plan.execute(state1.data, inverse=True)
		if state2 is not None:
			self._plan.execute(state2.data, inverse=True)

	def _create(self, two_component=False, comp=0, ratio=0.5,
			precision=1e-6, verbose=True):

		assert not two_component or comp == COMP_1_minus1

		desired_N1 = self._constants.N * ratio
		desired_N2 = self._constants.N * (1 - ratio)

		if two_component:
			# it would be nice to use two-component TF state here,
			# but the formula is quite complex, and it is much easier
			# just to start from uniform distribution
			# (not two one-component TF-states, because in case of
			# immiscible regime they are far from ground state)

			state1 = State(self._env, self._constants, comp=comp)
			state1.fillWithOnes()

			state2 = State(self._env, self._constants, comp=COMP_2_1)
			state2.fillWithOnes()
		else:
			# TF state is a good first approximation in case of one-component cloud
			state1 = self._tf_gs.create(comp=comp, N=self._constants.N)
			state2 = None

		stats = self._statistics
		E = 0

		if two_component:
			new_E = stats.countEnergyTwoComponent(state1, state2)
		else:
			new_E = stats.countEnergy(state1)

		self._toKSpace(state1, state2)

		while abs(E - new_E) / new_E > precision:

			# propagation
			self._kpropagate(state1, state2)
			self._toXSpace(state1, state2)
			self._xpropagate(state1, state2)
			self._toKSpace(state1, state2)
			self._kpropagate(state1, state2)

			# normalization

			self._toXSpace(state1, state2)

			# renormalize
			if two_component:
				N1 = stats.countParticles(state1)
				N2 = stats.countParticles(state2)
				c1 = math.sqrt(desired_N1 / N1)
				c2 = math.sqrt(desired_N2 / N2)
				self._renormalize(state1, state2, (c1, c2))
			else:
				N = stats.countParticles(state1)
				self._renormalize(state1, state2, math.sqrt(self._constants.N / N))

			E = new_E
			if two_component:
				new_E = stats.countEnergyTwoComponent(state1, state2, N=self._constants.N)
			else:
				new_E = stats.countEnergy(state1, N=self._constants.N)

			self._toKSpace(state1, state2)

		self._toXSpace(state1, state2)

		if verbose:
			if two_component:
				print "Ground state calculation (two components):" + \
					" N = " + str(stats.countParticles(state1)) + \
						" + " + str(stats.countParticles(state2)) + \
					" E = " + str(stats.countEnergyTwoComponent(state1, state2)) + \
					" mu = " + str(stats.countMuTwoComponent(state1, state2))
			else:
				print "Ground state calculation (one component):" + \
					" N = " + str(stats.countParticles(state1)) + \
					" E = " + str(stats.countEnergy(state1)) + \
					" mu = " + str(stats.countMu(state1))

		return state1, state2

	def createCloud(self, two_component=False, ratio=0.5, precision=1e-6):
		state1, state2 = self._create(two_component=two_component, ratio=ratio, precision=precision)
		return TwoComponentCloud(self._env, self._constants, a=state1, b=state2)

	def createState(self, comp=0, precision=1e-6):
		state1, state2 = self._create(two_component=False, comp=comp, precision=precision)
		return state1
