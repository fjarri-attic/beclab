"""
Different meters for particle states (measuring particles number, energy and so on)
"""

import math

from .globals import *
from .fft import createPlan
from .reduce import getReduce
from .constants import *


class State(PairedCalculation):

	def __init__(self, env, constants, type=PSI_FUNC, comp=COMP_1_minus1):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants
		self.type = type
		self.shape = constants.shape if type == PSI_FUNC else constants.ens_shape
		self.size = self.shape[0] * self.shape[1] * self.shape[2]
		self.dtype = constants.complex.dtype
		self.comp = comp

		self._prepare()
		self._initializeMemory()

	def copy(self):
		res = State(self._env, self._constants, type=self.type, comp=self.comp)
		res.data = self._env.copyBuffer(self.data)
		return res

	def _cpu__prepare(self):
		pass

	def _cpu__initializeMemory(self):
		self.data = numpy.zeros(self.shape, dtype=self.dtype)

	def _cpu_fillWithOnes(self):
		self.data = numpy.ones(self.shape, dtype=self.dtype)

	def _gpu__prepare(self):
		kernel_template = """
			<%!
				from math import sqrt
			%>

			__kernel void fillWithZeroes(__global ${c.complex.name} *res)
			{
				DEFINE_INDEXES;
				res[index] = ${c.complex.ctr}(0, 0);
			}

			__kernel void fillWithOnes(__global ${c.complex.name} *res)
			{
				DEFINE_INDEXES;
				res[index] = ${c.complex.ctr}(1, 0);
			}

			// Initialize ensembles with steady state + noise for Wigner quasiprobability function
			__kernel void initializeEnsembles(__global ${c.complex.name} *wigner_func,
				__global ${c.complex.name} *psi_func, __global ${c.complex.name} *randoms)
			{
				DEFINE_INDEXES;
				${c.complex.name} psi_val = psi_func[index % ${c.cells}];
				${c.scalar.name} coeff = (${c.scalar.name})${1.0 / sqrt(c.dV)};

				wigner_func[index] = psi_val + complex_mul_scalar(randoms[index], coeff);
			}
		"""

		self._program = self._env.compile(kernel_template, self._constants)
		self._fillWithZeroes = self._program.fillWithZeroes
		self._fillWithOnes = self._program.fillWithOnes
		self._initializeEnsembles = self._program.initializeEnsembles

	def _gpu__initializeMemory(self):
		self.data = self._env.allocate(self.shape, self.dtype)
		self._fillWithZeroes(self.shape, self.data)

	def _gpu_fillWithOnes(self):
		self._fillWithOnes(self.shape, self.data)

	def _gpu__toWigner(self, new_data, randoms):
		randoms_gpu = self._env.allocate(randoms.shape, randoms.dtype)
		cl.enqueue_write_buffer(self._env.queue, randoms_gpu, randoms)

		self._initializeEnsembles(self._constants.ens_shape, new_data, self.data, randoms_gpu)

	def _cpu__toWigner(self, new_data, randoms):
		coeff = 1.0 / math.sqrt(self._constants.dV)
		size = self._constants.cells * self._constants.ensembles

		for e in range(self._constants.ensembles):
			start = e * self._constants.nvz
			stop = (e + 1) * self._constants.nvz

			new_data[start:stop,:,:] = self.data + randoms[start:stop,:,:] * coeff

	def toWigner(self):

		assert self.type == PSI_FUNC

		new_data = self._env.allocate(self._constants.ens_shape, self._constants.complex.dtype)

		randoms = (numpy.random.normal(scale=0.5, size=self._constants.ens_shape) +
			1j * numpy.random.normal(scale=0.5, size=self._constants.ens_shape)).astype(self._constants.complex.dtype)

		self._toWigner(new_data, randoms)
		self.data = new_data
		self.shape = self._constants.ens_shape
		self.size = self._constants.cells * self._constants.ensembles

		self.type = WIGNER


class TwoComponentCloud:

	def __init__(self, env, constants, a=None, b=None):
		assert a is not None or b is not None
		assert a is None or b is None or a.type == b.type
		assert a is None or a.comp == COMP_1_minus1
		assert b is None or b.comp == COMP_2_1

		self._env = env
		self._constants = constants

		if a is None:
			type = b.type
		else:
			type = a.type

		self.type = type

		self.time = 0.0

		self.a = a.copy() if a is not None else State(env, constants, type=type, comp=COMP_1_minus1)
		self.b = b.copy() if b is not None else State(env, constants, type=type, comp=COMP_2_1)

	def toWigner(self):
		self.a.toWigner()
		self.b.toWigner()
		self.type = self.a.type

	def copy(self):
		res = TwoComponentCloud(self._env, self._constants, a=self.a, b=self.b)
		res.time = self.time
		return res


class ParticleStatistics(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)
		self._reduce = getReduce(env, constants)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _cpu_getAverageDensity(self, state):
		abs_values = numpy.abs(state.data)
		normalized_values = abs_values * abs_values

		if state.type == WIGNER:
			density = self._reduce.sparse(normalized_values, self._constants.cells)
			density /= self._constants.ensembles
			return density.reshape(self._constants.shape) - 0.5 / self._constants.dV
		else:
			return normalized_values

	def _cpu__countState(self, state, coeff, N):
		kdata = self._env.allocate(state.shape, dtype=state.dtype)
		res = self._env.allocate(state.shape, dtype=self._constants.scalar.dtype)

		data = state.data
		batch = state.size / self._constants.cells

		self._plan.execute(data, kdata, inverse=True, batch=batch)

		n = numpy.abs(data) ** 2
		xk = data * kdata

		g = self._constants.g[(state.comp, state.comp)]

		for e in xrange(batch):
			start = e * self._constants.cells
			stop = (e + 1) * self._constants.cells
			res[start:stop,:,:] = numpy.abs(n[start:stop,:,:] * (self._potentials +
				n[start:stop,:,:] * (g / coeff)) +
				xk[start:stop,:,:] * self._kvectors)

		return self._reduce(res) / batch * self._constants.dV / N

	def _cpu__countStateTwoComponent(self, state, second_state, coeff, N):
		n = numpy.abs(state.data) ** 2
		second_n = numpy.abs(second_state.data) ** 2

		g = self._constants.g[(state.comp, state.comp)]
		interaction_g = self._constants.g[(state.comp, second_state.comp)]

		batch = state.size / self._constants.cells

		data = state.data
		second_data = second_state.data
		kdata = self._env.allocate(state.shape, dtype=state.dtype)
		res = self._env.allocate(state.shape, dtype=self._constants.scalar.dtype)

		self._plan.execute(data, kdata, inverse=True, batch=batch)

		xk = data.conj() * kdata
		for e in xrange(batch):
			start = e * self._constants.cells
			stop = (e + 1) * self._constants.cells
			res[start:stop,:,:] = numpy.abs(n[start:stop,:,:] * (self._potentials +
				n[start:stop,:,:] * (g / coeff) +
				second_n[start:stop,:,:] * (interaction_g / coeff)) +
				xk[start:stop,:,:] * self._kvectors)

		return self._reduce(res) / batch * self._constants.dV / N

	def _cpu_getVisibility(self, state1, state2):
		ensembles = state1.size / self._constants.cells
		N1 = self.countParticles(state1)
		N2 = self.countParticles(state2)

		coeff = self._constants.dV / ensembles
		interaction = abs(self._reduce(state1.data * state2.data.conj())) * coeff

		return 2 * interaction / (N1 + N2)

	def _gpu__prepare(self):
		kernel_template = """
			__kernel void calculateDensity(__global ${c.scalar.name} *res,
				__global ${c.complex.name} *state, int ensembles,
				${c.scalar.name} statistics_term)
			{
				DEFINE_INDEXES;
				res[index] = (squared_abs(state[index]) + statistics_term) / ensembles;
			}

			__kernel void calculateDensity2(__global ${c.scalar.name} *a_res,
				__global ${c.scalar.name} *b_res, __global ${c.complex.name} *a_state,
				__global ${c.complex.name} *b_state, ${c.scalar.name} statistics_term)
			{
				DEFINE_INDEXES;
				a_res[index] = squared_abs(a_state[index]) + statistics_term;
				b_res[index] = squared_abs(b_state[index]) + statistics_term;
			}

			__kernel void calculateInteraction(__global ${c.complex.name} *interaction,
				__global ${c.complex.name} *a_state, __global ${c.complex.name} *b_state)
			{
				DEFINE_INDEXES;
				interaction[index] = complex_mul(a_state[index], conj(b_state[index]));
			}

			%for name, coeff in (('Energy', 2), ('Mu', 1)):
				__kernel void calculate${name}(__global ${c.scalar.name} *res,
					__global ${c.complex.name} *xstate, __global ${c.complex.name} *kstate,
					read_only image3d_t potentials, read_only image3d_t kvectors,
					${c.scalar.name} g)
				{
					DEFINE_INDEXES;

					${c.scalar.name} potential = get_float_from_image(potentials, i, j, k);
					${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

					${c.scalar.name} n = squared_abs(xstate[index]);
					${c.complex.name} differential =
						complex_mul(complex_mul(xstate[index], kstate[index]), kvector);
					${c.scalar.name} nonlinear = n * (potential + g * n / ${coeff});

					// differential.y will be equal to 0, because \psi * D \psi is a real number
					res[index] = nonlinear + differential.x;
				}

				__kernel void calculate${name}2(__global ${c.scalar.name} *res,
					__global ${c.complex.name} *xstate1, __global ${c.complex.name} *kstate1,
					__global ${c.complex.name} *xstate2, __global ${c.complex.name} *kstate2,
					read_only image3d_t potentials, read_only image3d_t kvectors,
					${c.scalar.name} g11, ${c.scalar.name} g22, ${c.scalar.name} g12)
				{
					DEFINE_INDEXES;

					${c.scalar.name} potential = get_float_from_image(potentials, i, j, k);
					${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

					${c.scalar.name} n1 = squared_abs(xstate1[index]);
					${c.scalar.name} n2 = squared_abs(xstate2[index]);

					${c.complex.name} differential1 =
						complex_mul(complex_mul(xstate1[index], kstate1[index]), kvector);
					${c.complex.name} differential2 =
						complex_mul(complex_mul(xstate2[index], kstate2[index]), kvector);

					${c.scalar.name} nonlinear1 = n1 * (potential +
						(${c.scalar.name})${c.g11} * n1 / ${coeff} +
						(${c.scalar.name})${c.g12} * n2 / ${coeff});
					${c.scalar.name} nonlinear2 = n2 * (potential +
						(${c.scalar.name})${c.g12} * n1 / ${coeff} +
						(${c.scalar.name})${c.g22} * n2 / ${coeff});

					// differential.y will be equal to 0, because \psi * D \psi is a real number
					res[index] = nonlinear1 + differential1.x +
						nonlinear2 + differential2.x;
				}
			%endfor
		"""

		self._program = self._env.compile(kernel_template, self._constants)

		self._calculateMu = self._program.calculateMu
		self._caclculateEnergy = self._program.calculateEnergy
		self._calculateDensity = self._program.calculateDensity
		self._calculateDensity2 = self._program.calculateDensity2
		self._calculateInteraction = self._program.calculateInteraction

		self._calculateMu2 = self._program.calculateMu2
		self._caclculateEnergy2 = self._program.calculateEnergy2

	def _gpu_getAverageDensity(self, state):
		density = self._env.allocate(state.shape, self._constants.scalar.dtype)
		ensembles = state.size / self._constants.cells

		statistics_term = 0 if state.type == PSI_FUNC else -0.5 / self._constants.dV
		self._calculateDensity(state.shape, density, state.data,
			numpy.int32(ensembles), self._constants.scalar.cast(statistics_term))
		density = self._reduce.sparse(density, final_length=self._constants.cells)
		return density.reshape(self._constants.shape)

	def _gpu__countState(self, state, coeff, N):
		if coeff == 1:
			func = self._calculateMu
		else:
			func = self._caclculateEnergy

		kstate = self._env.allocate(state.shape, dtype=state.dtype)
		res = self._env.allocate(state.shape, dtype=self._constants.scalar.dtype)
		self._plan.execute(state.data, kstate, inverse=True, batch=state.size / self._constants.cells)
		func(state.shape, res, state.data, kstate, self._potentials, self._kvectors,
			self._constants.g[(state.comp, state.comp)])
		return self._reduce(res) / (state.size / self._constants.cells) * self._constants.dV / N

	def _gpu__countStateTwoComponent(self, state1, state2, coeff, N):
		kstate1 = self._env.allocate(state1.shape, dtype=self._constants.complex.dtype)
		kstate2 = self._env.allocate(state2.shape, dtype=self._constants.complex.dtype)
		res = self._env.allocate(state1.shape, dtype=self._constants.scalar.dtype)

		self._plan.execute(state1.data, kstate1, inverse=True, batch=state1.size / self._constants.cells)
		self._plan.execute(state2.data, kstate2, inverse=True, batch=state2.size / self._constants.cells)

		g = self._constants.g
		g11 = g[(state1.comp, state1.comp)]
		g22 = g[(state2.comp, state2.comp)]
		g12 = g[(state1.comp, state2.comp)]

		if coeff == 1:
			func = self._calculateMu2
		else:
			func = self._caclculateEnergy2

		func(state1.shape, res, state1.data, kstate1,
			state2.data, kstate2, self._potentials, self._kvectors, g11, g22, g12)
		return self._reduce(res) / (state1.size / self._constants.cells) * self._constants.dV / N

	def _gpu_getVisibility(self, state1, state2):
		density1 = self._env.allocate(state1.shape, self._constants.scalar.dtype)
		density2 = self._env.allocate(state2.shape, self._constants.scalar.dtype)

		interaction = self._env.allocate(state1.shape, self._constants.complex.dtype)

		self._calculateInteraction(state1.shape, interaction, state1.data, state2.data)

		N1 = self.countParticles(state1)
		N2 = self.countParticles(state2)

		coeff = self._constants.dV / (state1.size / self._constants.cells)
		interaction = self._reduce(interaction) * coeff

		return 2 * abs(interaction) / (N1 + N2)

	def countParticles(self, state):
		return self._reduce(self.getAverageDensity(state)) * self._constants.dV

	def _countStateGeneric(self, state, coeff, N):
		# TODO: work out the correct formula for Wigner function's E/mu
		if state.type != PSI_FUNC:
			raise NotImplementedError()

		if N is None:
			N = self.countParticles(state)

		return self._countState(state, coeff, N)


	def countEnergy(self, state, N=None):
		return self._countStateGeneric(state, 2, N)

	def countMu(self, state, N=None):
		return self._countStateGeneric(state, 1, N)

	def _countTwoComponentGeneric(self, state1, state2, coeff, N):
		# TODO: work out the correct formula for Wigner function's E/mu
		if state1.type != PSI_FUNC or state2.type != PSI_FUNC:
			raise NotImplementedError()

		if N is None:
			N = self.countParticles(state1) + self.countParticles(state2)

		return self._countStateTwoComponent(state1, state2, coeff, N) + \
			self._countStateTwoComponent(state2, state1, coeff, N)

	def countEnergyTwoComponent(self, state1, state2, N=None):
		return self._countTwoComponentGeneric(state1, state2, 2, N)

	def countMuTwoComponent(self, state1, state2, N=None):
		return self._countTwoComponentGeneric(state1, state2, 1, N)


class BlochSphereProjection(PairedCalculation):

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

	def _cpu_getProjection(self, a, b, amp_points, phase_points, amp_range, phase_range):

		if a.type != PSI_FUNC or b.type != PSI_FUNC:
			raise NotImplementedError()

		amp_min, amp_max = amp_range
		phase_min, phase_max = phase_range

		res = numpy.zeros((amp_points, phase_points), dtype=self._constants.scalar.dtype)

		a = a.data.ravel()
		b = b.data.ravel()

		amp_a = numpy.abs(a)
		amp_b = numpy.abs(b)
		phase_a = numpy.angle(a)
		phase_b = numpy.angle(b)

		density = amp_a * amp_a + amp_b * amp_b
		density = density / numpy.sum(density)

		d_amp = (amp_max - amp_min) / (amp_points - 1)
		d_phase = (phase_max - phase_min) / (phase_points - 1)

		phase_diff = phase_b - phase_a
		for i in xrange(a.size):
			if phase_diff[i] < 0:
				phase_diff[i] += 2 * math.pi

		amp_diff = ((2 * numpy.arctan(amp_b / amp_a) - amp_min) / d_amp).astype(numpy.int32)
		phase_diff = ((phase_diff - phase_min) / d_phase).astype(numpy.int32)

		for i in xrange(a.size):
			amp_coord = amp_diff[i]
			phase_coord = phase_diff[i]

			if amp_coord < 0 or amp_coord >= amp_points or phase_coord < 0 or phase_coord >= phase_points:
				continue

			res[amp_coord, phase_coord] += density[i]

		return res

	def getAverages(self, a, b):

		if a.type != PSI_FUNC or b.type != PSI_FUNC:
			raise NotImplementedError()

		a = a.data
		b = b.data

		amp_a = numpy.abs(a)
		amp_b = numpy.abs(b)
		density = amp_a * amp_a + amp_b * amp_b
		density_total = numpy.sum(density)

		phase_a = numpy.angle(a)
		phase_b = numpy.angle(b)

		max_a = numpy.max(amp_a)
		max_b = numpy.max(amp_b)

		avg_phase = numpy.sum((phase_b - phase_a) * density) / density_total
		avg_amp = numpy.sum(2 * numpy.arctan(amp_b / amp_a) * density) / density_total

		if avg_phase < 0:
			avg_phase -= (int(avg_phase / (math.pi * 2.0)) - 1) * math.pi * 2.0
		elif avg_phase > 2.0 * math.pi:
			avg_phase -= int(avg_phase / (math.pi * 2.0)) * math.pi * 2.0

		return avg_amp, avg_phase


class Projection:

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants
		self._reduce = getReduce(env, constants)
		self._stats = ParticleStatistics(env, constants)

	def getXY(self, state):
		density = self._stats.getAverageDensity(state)
		x = self._constants.nvx
		y = self._constants.nvy
		return self._env.toCPU(self._reduce.sparse(density, final_length=x * y), shape=(y, x))

	def getYZ(self, state):
		density = self._stats.getAverageDensity(state)
		y = self._constants.nvy
		z = self._constants.nvz
		return self._env.toCPU(self._reduce(density, final_length=y * z), shape=(z, y))

	def getZ(self, state):
		density = self._stats.getAverageDensity(state)
		z = self._constants.nvz
		return self._env.toCPU(self._reduce(density, final_length=z))


class Slice:

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants
		self._stats = ParticleStatistics(env, constants)

	def getXY(self, state):
		density = self._stats.getAverageDensity(state)
		temp = self._env.toCPU(density)
		return temp[self._constants.nvz / 2,:,:]

	def getYZ(self, state):
		density = self._stats.getAverageDensity(state)
		temp = self._env.toCPU(density).transpose((2, 0, 1))
		return temp[self._constants.nvx / 2,:,:]
