"""
Different meters for particle states (measuring particles number, energy and so on)
"""

import math

from .helpers import *
from .constants import *


class State(PairedCalculation):

	def __init__(self, env, constants, type=PSI_FUNC, comp=COMP_1_minus1, prepare=True):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants
		self.type = type
		self.shape = constants.shape if type == PSI_FUNC else constants.ens_shape
		self.size = constants.cells * (1 if type == PSI_FUNC else constants.ensembles)
		self.dtype = constants.complex.dtype
		self.comp = comp

		if prepare:
			self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
			self._prepare()
			self._initializeMemory()

	def copy(self, prepare=True):
		res = State(self._env, self._constants, type=self.type, comp=self.comp, prepare=prepare)
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

			EXPORTED_FUNC void fillWithZeroes(GLOBAL_MEM COMPLEX *res)
			{
				DEFINE_INDEXES;
				res[index] = complex_ctr(0, 0);
			}

			EXPORTED_FUNC void fillWithOnes(GLOBAL_MEM COMPLEX *res)
			{
				DEFINE_INDEXES;
				res[index] = complex_ctr(1, 0);
			}

			// Initialize ensembles with steady state
			EXPORTED_FUNC void initializeEnsembles(GLOBAL_MEM COMPLEX *wigner_func,
				GLOBAL_MEM COMPLEX *psi_func)
			{
				DEFINE_INDEXES;
				COMPLEX psi_val = psi_func[index % ${c.cells}];
				wigner_func[index] = psi_val;
			}

			EXPORTED_FUNC void fillEnsembles(GLOBAL_MEM COMPLEX *new_data,
				GLOBAL_MEM COMPLEX *data)
			{
				DEFINE_INDEXES;
				COMPLEX val = data[index];

				for(int i = 0; i < ${c.ensembles}; i++)
					new_data[index + i * ${c.cells}] = val;
			}

			EXPORTED_FUNC void addPlaneWaves(GLOBAL_MEM COMPLEX *kdata,
				GLOBAL_MEM COMPLEX *randoms,
				GLOBAL_MEM SCALAR *projector_mask)
			{
				DEFINE_INDEXES;
				SCALAR prj = projector_mask[cell_index];
				SCALAR coeff = ${1.0 / sqrt(c.V)};

				// add noise and remove high-energy components
				kdata[index] = kdata[index] + complex_mul_scalar(randoms[index], prj * coeff);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants)
		self._fillWithZeroes = self._program.fillWithZeroes
		self._fillWithOnes = self._program.fillWithOnes
		self._initializeEnsembles = self._program.initializeEnsembles
		self._addPlaneWaves = self._program.addPlaneWaves
		self._fillEnsembles = self._program.fillEnsembles

	def _gpu__initializeMemory(self):
		self.data = self._env.allocate(self.shape, self.dtype)
		self._fillWithZeroes(self.size, self.data)

	def _gpu_fillWithOnes(self):
		self._fillWithOnes(self.size, self.data)

	def _gpu__addVacuumParticles(self, randoms, mask):

		dtype = self.data.dtype
		batch = self.data.size / self._constants.cells

		randoms = self._env.toDevice(randoms)

		kdata = self._env.allocate(self._constants.ens_shape, dtype=dtype)
		self._plan.execute(self.data, kdata, inverse=True, batch=batch)
		self._addPlaneWaves(kdata.size, kdata, randoms, mask)
		self._plan.execute(kdata, self.data, batch=batch)

	def _cpu__addVacuumParticles(self, randoms, mask):

		coeff = 1.0 / math.sqrt(self._constants.V)

		dtype = self.data.dtype
		batch = self.data.size / self._constants.cells
		nvz = self._constants.nvz

		kdata = self._env.allocate(self._constants.ens_shape, dtype=dtype)

		self._plan.execute(self.data, kdata, inverse=True, batch=batch)

		for e in xrange(batch):
			start = e * nvz
			stop = (e + 1) * nvz

			if self._constants.dim == 3:
				kdata[start:stop,:,:] += mask * coeff * randoms[start:stop,:,:]
				kdata[start:stop,:,:] *= mask # remove high-energy components
			else:
				kdata[start:stop] += mask * coeff * randoms[start:stop]
				kdata[start:stop] *= mask # remove high-energy components

		self._plan.execute(kdata, self.data, batch=batch)

	def toWigner(self):

		assert self.type == PSI_FUNC

		self.createEnsembles()

		randoms = (numpy.random.normal(scale=0.5, size=self._constants.ens_shape) +
			1j * numpy.random.normal(scale=0.5, size=self._constants.ens_shape)).astype(self._constants.complex.dtype)

		projector_mask, _ = getProjectorMask(self._env, self._constants)
		self._addVacuumParticles(randoms, projector_mask)

		self.type = WIGNER

	def _cpu__createEnsembles(self):
		tile = tuple([self._constants.ensembles] + [1] * (self._constants.dim - 1))
		return numpy.tile(self.data, tile)

	def _gpu__createEnsembles(self):
		new_data = self._env.allocate(self._constants.ens_shape, self._constants.complex.dtype)
		self._fillEnsembles(self._constants.cells, new_data, self.data)
		return new_data

	def createEnsembles(self):

		assert self.size == self._constants.cells

		self.data = self._createEnsembles()
		self.shape = self._constants.ens_shape
		self.size = self._constants.cells * self._constants.ensembles


class TwoComponentCloud:

	def __init__(self, env, constants, a=None, b=None, prepare=True):
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

		self.a = a.copy(prepare=prepare) \
			if a is not None else State(env, constants, type=type, comp=COMP_1_minus1)
		self.b = b.copy(prepare=prepare) \
			if b is not None else State(env, constants, type=type, comp=COMP_2_1)

	def toWigner(self):
		self.a.toWigner()
		self.b.toWigner()
		self.type = self.a.type

	def copy(self, prepare=True):
		res = TwoComponentCloud(self._env, self._constants,
			a=self.a, b=self.b, prepare=prepare)
		res.time = self.time
		return res

	def createEnsembles(self):
		self.a.createEnsembles()
		self.b.createEnsembles()


class ParticleStatistics(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
		self._reduce = createReduce(env, constants.scalar.dtype)
		self._creduce = createReduce(env, constants.complex.dtype)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		self._projector_mask, self._projector_modes = getProjectorMask(self._env, self._constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _cpu_getAverageDensity(self, state):
		normalized_values = numpy.abs(state.data) ** 2

		if state.type == WIGNER:
			# What we are returning here is not in fact the measurable density.
			# In order to return density, we would have to calculate integral
			# of \delta_P for each cell, which is not that simple in general case
			# (although, it is simple for plane wave basis).
			# So, we are just returning reduced data for countParticles(),
			# which will subtract vacuum particles (which is a lot simpler)

			density = self._reduce.sparse(normalized_values, self._constants.cells)
			return density.reshape(self._constants.shape) / self._constants.ensembles
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

		g_by_hbar = self._constants.g_by_hbar[(state.comp, state.comp)]

		if self._constants.dim == 1:
			for e in xrange(batch):
				start = e * self._constants.cells
				stop = (e + 1) * self._constants.cells
				res[start:stop] = numpy.abs(n[start:stop] * (self._potentials +
					n[start:stop] * (g_by_hbar / coeff)) +
					xk[start:stop] * self._kvectors)

		else:
			for e in xrange(batch):
				start = e * self._constants.cells
				stop = (e + 1) * self._constants.cells
				res[start:stop,:,:] = numpy.abs(n[start:stop,:,:] * (self._potentials +
					n[start:stop,:,:] * (g_by_hbar / coeff)) +
					xk[start:stop,:,:] * self._kvectors)

		return self._reduce(res) / batch * self._constants.dV / N * self._constants.hbar

	def _cpu__countStateTwoComponent(self, state, second_state, coeff, N):
		n = numpy.abs(state.data) ** 2
		second_n = numpy.abs(second_state.data) ** 2

		g_by_hbar = self._constants.g_by_hbar[(state.comp, state.comp)]
		interaction_g_by_hbar = self._constants.g_by_hbar[(state.comp, second_state.comp)]

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
				n[start:stop,:,:] * (g_by_hbar / coeff) +
				second_n[start:stop,:,:] * (interaction_g_by_hbar / coeff)) +
				xk[start:stop,:,:] * self._kvectors)

		return self._reduce(res) / batch * self._constants.dV / N * self._constants.hbar

	def _cpu_getVisibility(self, state1, state2):
		ensembles = state1.size / self._constants.cells
		N1 = self.countParticles(state1)
		N2 = self.countParticles(state2)

		coeff = self._constants.dV / ensembles
		interaction = abs(self._creduce(state1.data * state2.data.conj())) * coeff

		return 2 * interaction / (N1 + N2)

	def _gpu__prepare(self):
		kernel_template = """
			EXPORTED_FUNC void calculateDensity(GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *state, int ensembles)
			{
				DEFINE_INDEXES;
				res[index] = squared_abs(state[index]) / ensembles;
			}

			EXPORTED_FUNC void calculateInteraction(GLOBAL_MEM COMPLEX *interaction,
				GLOBAL_MEM COMPLEX *a_state, GLOBAL_MEM COMPLEX *b_state)
			{
				DEFINE_INDEXES;
				interaction[index] = complex_mul(a_state[index], conj(b_state[index]));
			}

			%for name, coeff in (('Energy', 2), ('Mu', 1)):
				EXPORTED_FUNC void calculate${name}(GLOBAL_MEM SCALAR *res,
					GLOBAL_MEM COMPLEX *xstate, GLOBAL_MEM COMPLEX *kstate,
					GLOBAL_MEM SCALAR *potentials, GLOBAL_MEM SCALAR *kvectors,
					SCALAR g_by_hbar)
				{
					DEFINE_INDEXES;

					SCALAR potential = potentials[cell_index];
					SCALAR kvector = kvectors[cell_index];

					SCALAR n = squared_abs(xstate[index]);
					COMPLEX differential =
						complex_mul_scalar(complex_mul(xstate[index], kstate[index]), kvector);
					SCALAR nonlinear = n * (potential + g_by_hbar * n / ${coeff});

					// differential.y will be equal to 0, because \psi * D \psi is a real number
					res[index] = nonlinear + differential.x;
				}

				EXPORTED_FUNC void calculate${name}2(GLOBAL_MEM SCALAR *res,
					GLOBAL_MEM COMPLEX *xstate1, GLOBAL_MEM COMPLEX *kstate1,
					GLOBAL_MEM COMPLEX *xstate2, GLOBAL_MEM COMPLEX *kstate2,
					GLOBAL_MEM SCALAR *potentials, GLOBAL_MEM SCALAR *kvectors,
					SCALAR g11_by_hbar, SCALAR g22_by_hbar,
					SCALAR g12_by_hbar)
				{
					DEFINE_INDEXES;

					SCALAR potential = potentials[cell_index];
					SCALAR kvector = kvectors[cell_index];

					SCALAR n1 = squared_abs(xstate1[index]);
					SCALAR n2 = squared_abs(xstate2[index]);

					COMPLEX differential1 =
						complex_mul_scalar(complex_mul(xstate1[index], kstate1[index]), kvector);
					COMPLEX differential2 =
						complex_mul_scalar(complex_mul(xstate2[index], kstate2[index]), kvector);

					SCALAR nonlinear1 = n1 * (potential +
						g11_by_hbar * n1 / ${coeff} +
						g12_by_hbar * n2 / ${coeff});
					SCALAR nonlinear2 = n2 * (potential +
						g12_by_hbar * n1 / ${coeff} +
						g22_by_hbar * n2 / ${coeff});

					// differential.y will be equal to 0, because \psi * D \psi is a real number
					res[index] = nonlinear1 + differential1.x +
						nonlinear2 + differential2.x;
				}
			%endfor
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants)

		self._calculateMu = self._program.calculateMu
		self._calculateEnergy = self._program.calculateEnergy
		self._calculateDensity = self._program.calculateDensity
		self._calculateInteraction = self._program.calculateInteraction

		self._calculateMu2 = self._program.calculateMu2
		self._calculateEnergy2 = self._program.calculateEnergy2

	def _gpu_getAverageDensity(self, state):
		density = self._env.allocate(state.shape, self._constants.scalar.dtype)
		ensembles = state.size / self._constants.cells

		# This function does not return "real" density for Wigner case.
		# See comment for CPU version for details.
		self._calculateDensity(state.size, density, state.data, numpy.int32(ensembles))
		density = self._reduce.sparse(density, final_length=self._constants.cells)
		density.shape = self._constants.shape
		return density

	def _gpu__countState(self, state, coeff, N):
		if coeff == 1:
			func = self._calculateMu
		else:
			func = self._calculateEnergy

		kstate = self._env.allocate(state.shape, dtype=state.dtype)
		res = self._env.allocate(state.shape, dtype=self._constants.scalar.dtype)
		self._plan.execute(state.data, kstate, inverse=True, batch=state.size / self._constants.cells)
		func(state.size, res, state.data, kstate, self._potentials, self._kvectors,
			self._constants.scalar.cast(self._constants.g_by_hbar[(state.comp, state.comp)]))
		return self._reduce(res) / (state.size / self._constants.cells) * \
			self._constants.dV / N * self._constants.hbar

	def _gpu__countStateTwoComponent(self, state1, state2, coeff, N):
		kstate1 = self._env.allocate(state1.shape, dtype=self._constants.complex.dtype)
		kstate2 = self._env.allocate(state2.shape, dtype=self._constants.complex.dtype)
		res = self._env.allocate(state1.shape, dtype=self._constants.scalar.dtype)

		self._plan.execute(state1.data, kstate1, inverse=True, batch=state1.size / self._constants.cells)
		self._plan.execute(state2.data, kstate2, inverse=True, batch=state2.size / self._constants.cells)

		g_by_hbar = self._constants.g_by_hbar
		cast = self._constants.scalar.cast
		g11_by_hbar = cast(g_by_hbar[(state1.comp, state1.comp)])
		g22_by_hbar = cast(g_by_hbar[(state2.comp, state2.comp)])
		g12_by_hbar = cast(g_by_hbar[(state1.comp, state2.comp)])

		if coeff == 1:
			func = self._calculateMu2
		else:
			func = self._calculateEnergy2

		func(state1.size, res, state1.data, kstate1,
			state2.data, kstate2, self._potentials, self._kvectors,
				g11_by_hbar, g22_by_hbar, g12_by_hbar)

		return self._reduce(res) / (state1.size / self._constants.cells) * \
			self._constants.dV / N * self._constants.hbar

	def _gpu_getVisibility(self, state1, state2):
		interaction = self._env.allocate(state1.shape, self._constants.complex.dtype)
		self._calculateInteraction(state1.size, interaction, state1.data, state2.data)

		N1 = self.countParticles(state1)
		N2 = self.countParticles(state2)

		coeff = self._constants.dV / (state1.size / self._constants.cells)
		interaction = self._creduce(interaction) * coeff

		return 2 * abs(interaction) / (N1 + N2)

	def _gpu__getEnsembleData(self, state1, state2):
		interaction = self._env.allocate(state1.shape, self._constants.complex.dtype)
		self._calculateInteraction(state1.size, interaction, state1.data, state2.data)

		n1 = self._env.allocate(state1.shape, self._constants.scalar.dtype)
		n2 = self._env.allocate(state2.shape, self._constants.scalar.dtype)

		self._calculateDensity(state1.size, n1, state1.data, numpy.int32(1))
		self._calculateDensity(state2.size, n2, state2.data, numpy.int32(1))

		return interaction, n1, n2

	def _cpu__getEnsembleData(self, state1, state2):
		return state1.data * state2.data.conj(), numpy.abs(state1.data) ** 2, \
			numpy.abs(state2.data) ** 2

	def getPhaseNoise(self, state1, state2):
		ensembles = state1.size / self._constants.cells
		get = self._env.fromDevice
		reduce = self._reduce
		creduce = self._creduce
		dV = self._constants.dV

		i, n1, n2 = self._getEnsembleData(state1, state2)

		n1 = get(reduce(n1, ensembles)) * dV
		n2 = get(reduce(n2, ensembles)) * dV

		if state1.type == WIGNER:
			n1 -= self._projector_modes / 2
			n2 -= self._projector_modes / 2

		i = get(creduce(i, ensembles))

		Pperp = 2.0 * i / (n1 + n2)
		Pperp /= numpy.abs(Pperp)

		return ((Pperp * (Pperp.mean()).conj()).imag).std()

	def countParticles(self, state):
		N = self._reduce(self.getAverageDensity(state)) * self._constants.dV
		if state.type == WIGNER:
			# Since returned density is not "real" density for Wigner case,
			# we have to subtract vacuum particles. See comment in
			# getAverageDensity() for details.
			return N - self._projector_modes / 2
		else:
			return N

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

		if self._env.gpu:
			return self._countStateTwoComponent(state1, state2, coeff, N)
		else:
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
		self._reduce = createReduce(env, constants.scalar.dtype)
		self._stats = ParticleStatistics(env, constants)

	def getXY(self, state):
		density = self._stats.getAverageDensity(state)
		x = self._constants.nvx
		y = self._constants.nvy
		return self._env.fromDevice(self._reduce.sparse(density, final_length=x * y), shape=(y, x))

	def getYZ(self, state):
		density = self._stats.getAverageDensity(state)
		y = self._constants.nvy
		z = self._constants.nvz
		return self._env.fromDevice(self._reduce(density, final_length=y * z), shape=(z, y))

	def getZ(self, state):
		density = self._stats.getAverageDensity(state)
		z = self._constants.nvz
		return self._env.fromDevice(self._reduce(density, final_length=z))


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
