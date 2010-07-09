"""
Ground state calculation classes
"""

import math
import copy
import numpy

from .globals import *
from .fft import createPlan
from .state import ParticleStatistics, State, TwoComponentCloud
from .reduce import getReduce
from .constants import COMP_1_minus1, COMP_2_1


class TFGroundState(PairedCalculation):
	"""
	Ground state, calculated using Thomas-Fermi approximation
	(kinetic energy == 0)
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants
		self._potentials = getPotentials(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernel_template = """
			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			__kernel void fillWithTFGroundState(__global ${c.complex.name} *data,
				read_only image3d_t potentials, ${c.scalar.name} mu,
				${c.scalar.name} g)
			{
				DEFINE_INDEXES;

				float potential = get_float_from_image(potentials, i, j, k);

				${c.scalar.name} e = mu - potential;
				if(e > 0)
					data[index] = ${c.complex.ctr}(sqrt(e / g), 0);
				else
					data[index] = ${c.complex.ctr}(0, 0);
			}
		"""

		self._program = self._env.compile(kernel_template, self._constants)
		self._fillWithTFGroundState = self._program.fillWithTFGroundState

	def _gpu__create(self, data, g, mu):
		self._fillWithTFGroundState(data.shape, data, self._potentials, mu, g)

	def _cpu__create(self, data, g, mu):
		for i in xrange(self._constants.nvx):
			for j in xrange(self._constants.nvy):
				for k in xrange(self._constants.nvz):
					e = mu - self._potentials[k, j, i]
					data[k, j, i] = math.sqrt(max(e / g, 0))

	def create(self, comp=COMP_1_minus1, N=None):
		res = State(self._env, self._constants, comp=comp)

		g = self._constants.g[(comp, comp)]
		mu = self._constants.muTF(comp=comp, N=N)

		self._create(res.data, g, mu)
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
		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)
		self._statistics = ParticleStatistics(env, constants)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		self._k_coeff = numpy.exp(self._kvectors * (-self._constants.dt_steady / 2))

	def _gpu__prepare(self):
		kernel_template = """
			__kernel void multiply(__global ${c.complex.name} *data, ${c.scalar.name} coeff)
			{
				DEFINE_INDEXES;
				data[index] = complex_mul_scalar(data[index], coeff);
			}

			__kernel void multiply2(__global ${c.complex.name} *data1, __global ${c.complex.name} *data2,
				${c.scalar.name} c1, ${c.scalar.name} c2)
			{
				DEFINE_INDEXES;
				data1[index] = complex_mul_scalar(data1[index], c1);
				data2[index] = complex_mul_scalar(data2[index], c2);
			}

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			__kernel void propagateKSpace(__global ${c.complex.name} *data,
				read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

				${c.scalar.name} prop_coeff = native_exp(kvector *
					(${c.scalar.name})${-c.dt_steady / 2});
				${c.complex.name} temp = data[index];
				data[index] = complex_mul_scalar(temp, prop_coeff);
			}

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			// Version for processing two components at once
			__kernel void propagateKSpace2(
				__global ${c.complex.name} *data1, __global ${c.complex.name} *data2,
				read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

				${c.scalar.name} prop_coeff = native_exp(kvector *
					(${c.scalar.name})${-c.dt_steady / 2});

				data1[index] = complex_mul_scalar(data1[index], prop_coeff);
				data2[index] = complex_mul_scalar(data2[index], prop_coeff);
			}

			// Propagates state in x-space for steady state calculation
			__kernel void propagateXSpace(__global ${c.complex.name} *data,
				read_only image3d_t potentials, ${c.scalar.name} g)
			{
				DEFINE_INDEXES;

				${c.complex.name} a = data[index];

				//store initial x-space field
				${c.complex.name} a0 = a;

				${c.scalar.name} da;
				${c.scalar.name} V = get_float_from_image(potentials, i, j, k);

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					da = exp((${c.scalar.name})${c.dt_steady / 2} *
						(-V - g * squared_abs(a)));

					//propagate to midpoint using log derivative
					a = complex_mul_scalar(a0, da);
				%endfor

				//propagate to endpoint using log derivative
				data[index] = complex_mul_scalar(a, da);
			}

			// Propagates state in x-space for steady state calculation
			__kernel void propagateXSpace2(__global ${c.complex.name} *a,
				__global ${c.complex.name} *b, read_only image3d_t potentials,
				${c.scalar.name} g11, ${c.scalar.name} g22, ${c.scalar.name} g12)
			{
				DEFINE_INDEXES;

				${c.complex.name} a_res = a[index];
				${c.complex.name} b_res = b[index];

				//store initial x-space field
				${c.complex.name} a0 = a_res;
				${c.complex.name} b0 = b_res;

				${c.scalar.name} da, db, a_density, b_density;
				${c.scalar.name} V = get_float_from_image(potentials, i, j, k);

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					a_density = squared_abs(a_res);
					b_density = squared_abs(b_res);

					da = exp((${c.scalar.name})${c.dt_steady / 2} *
						(-V - g11 * a_density -	g12 * b_density));
					db = exp((${c.scalar.name})${c.dt_steady / 2} *
						(-V - g12 * a_density -	g22 * b_density));

					//propagate to midpoint using log derivative
					a_res = complex_mul_scalar(a0, da);
					b_res = complex_mul_scalar(b0, db);
				%endfor

				//propagate to endpoint using log derivative
				a[index] = complex_mul_scalar(a_res, da);
				b[index] = complex_mul_scalar(b_res, db);
			}
		"""

		self._program = self._env.compile(kernel_template, self._constants)

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
			self._propagateKSpace(state1.shape, state1.data, self._kvectors)
		else:
			self._propagateKSpace2(state1.shape, state1.data, state2.data, self._kvectors)

	def _cpu__xpropagate(self, state1, state2):
		p = self._potentials
		dt = -self._constants.dt_steady / 2

		if state2 is None:
			a0 = state1.data.copy()
			g = self._constants.g[(state1.comp, state1.comp)]

			for iter in xrange(self._constants.itmax):
				n = numpy.abs(state1.data) ** 2
				da = numpy.exp((p + n * g) * dt)
				state1.data = a0 * da
			state1.data *= da
		else:
			a0 = state1.data.copy()
			b0 = state2.data.copy()

			comp1 = state1.comp
			comp2 = state2.comp
			g = self._constants.g
			g11 = g[(comp1, comp1)]
			g12 = g[(comp1, comp2)]
			g22 = g[(comp2, comp2)]

			for iter in xrange(self._constants.itmax):
				na = numpy.abs(state1.data) ** 2
				nb = numpy.abs(state2.data) ** 2

				pa = p + na * g11 + nb * g12
				pb = p + nb * g22 + na * g12

				da = numpy.exp(pa * dt)
				db = numpy.exp(pb * dt)

				state1.data = a0 * da
				state2.data = b0 * db

			state1.data *= da
			state2.data *= db

	def _gpu__xpropagate(self, state1, state2):
		if state2 is None:
			g = self._constants.g[(state1.comp, state1.comp)]
			self._propagateXSpace(state1.shape, state1.data, self._potentials, g)
		else:
			comp1 = state1.comp
			comp2 = state2.comp
			g = self._constants.g
			g11 = g[(comp1, comp1)]
			g12 = g[(comp1, comp2)]
			g22 = g[(comp2, comp2)]

			self._propagateXSpace2(state1.shape, state1.data, state2.data,
				self._potentials, g11, g22, g12)

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
			self._multiply(state1.shape, state1.data, cast(coeff))
		else:
			c1, c2 = coeff
			self._multiply2(state1.shape, state1.data, state2.data, cast(c1), cast(c2))

	def _toXSpace(self, state1, state2):
		self._plan.execute(state1.data)
		if state2 is not None:
			self._plan.execute(state2.data)

	def _toKSpace(self, state1, state2):
		self._plan.execute(state1.data, inverse=True)
		if state2 is not None:
			self._plan.execute(state2.data, inverse=True)

	def _create(self, two_component=False, comp=COMP_1_minus1, ratio=0.5,
			precision=1e-6, verbose=True):

		assert not two_component or comp == COMP_1_minus1

		desired_N1 = self._constants.N * ratio
		desired_N2 = self._constants.N * (1 - ratio)

		#state1 = self._tf_gs.create(comp=comp, N=desired_N1)
		state1 = State(self._env, self._constants, comp=comp)
		state1.fillWithOnes()

		#state2 = self._tf_gs.create(comp=COMP_2_1, N=desired_N2) if two_component else None
		if two_component:
			state2 = State(self._env, self._constants, comp=COMP_2_1)
			state2.fillWithOnes()
		else:
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

	def createState(self, comp=COMP_1_minus1, precision=1e-6):
		state1, state2 = self._create(two_component=False, comp=comp, precision=precision)
		return state1
