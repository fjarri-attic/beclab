"""
Ground state calculation classes
"""

import numpy

from .helpers import *
from .wavefunction import WavefunctionSet
from .ground_state import ImaginaryTimeGroundState, RK5Propagation, Projector
from .evolution import Evolution, TerminateEvolution
from .constants import *


class EnergyCondition(PairedCalculation):

	def __init__(self, env, constants, grid, precision=0.5, E_modifier=0, N=None):
		PairedCalculation.__init__(self, env)
		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)
		self._previous_E = 0
		self._precision = precision
		self._N = N
		self._E_modifier = E_modifier
		self.times = []
		self.ediffs = []

	def __call__(self, t, dt, psi):
		if dt == 0:
			return

		E = psi.interaction_meter.getETotal().sum()
		E_diff = abs((self._previous_E - E) / (E + self._E_modifier)) / dt

		self.times.append(t)
		self.ediffs.append(E_diff)
		self._previous_E = E

		if E_diff < self._precision:
			raise TerminateEvolution()

	def getData(self):
		return numpy.array(self.times), numpy.array(self.ediffs)


class SOGroundStateEvo(Evolution):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		Evolution.__init__(self, env)
		self._constants = constants
		self._grid = grid
		self._projector = Projector(env, constants, grid)
		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL,
			dt=1e-5, itmax=3)
		self.prepare(**kwds)

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=self._p.ensembles)

		self._potentials = self._grid.potentials_device
		self._p.g = self._constants.g / self._constants.hbar
		energy = self._grid.energy
		if self._constants.so_coupling:
			self._mode_prop = self._env.toDevice(
				buildEnergyExp(energy, dt=self._p.dt / 2, imaginary_time=True))
		else:
			self._mode_prop = self._env.toDevice(numpy.exp(energy * (-self._p.dt / 2)))

	def _gpu__prepare_specific(self, **kwds):
		kernel_template = """
			// Propagates psi function in mode space
			%if c.so_coupling:
			EXPORTED_FUNC void mpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *mode_prop)
			{
				LIMITED_BY(gsize);

				COMPLEX mode_prop00 = mode_prop[GLOBAL_INDEX];
				COMPLEX mode_prop01 = mode_prop[GLOBAL_INDEX + ${g.msize}];
				COMPLEX mode_prop10 = mode_prop[GLOBAL_INDEX + ${g.msize * 2}];
				COMPLEX mode_prop11 = mode_prop[GLOBAL_INDEX + ${g.msize * 3}];

				COMPLEX data0 = data[GLOBAL_INDEX];
				COMPLEX data1 = data[GLOBAL_INDEX + ${g.msize}];

				data[GLOBAL_INDEX] = complex_mul(data0, mode_prop00) +
					complex_mul(data1, mode_prop01);
				data[GLOBAL_INDEX + ${g.msize}] = complex_mul(data0, mode_prop10) +
					complex_mul(data1, mode_prop11);
			}
			%else:
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
			%endif

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
		if self._constants.so_coupling:
			data_copy = data.copy()
			data[0] = mode_prop[0, 0] * data_copy[0] + mode_prop[0, 1] * data_copy[1]
			data[1] = mode_prop[1, 0] * data_copy[0] + mode_prop[1, 1] * data_copy[1]
		else:
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

	def _renormalize(self, psi, N):
		new_N = psi.density_meter.getNTotal()

		coeff = numpy.sqrt(N.sum() / new_N.sum())
		coeffs = [coeff] * self._p.components

		psi.multiplyBy(coeffs)

	def propagate(self, psi, t, max_dt):
		self._kernel_mpropagate(psi.size, psi.data, self._mode_prop)
		self._toMeasurementSpace(psi)
		self._kernel_xpropagate(psi.size, psi.data, self._potentials)
		self._toEvolutionSpace(psi)
		self._kernel_mpropagate(psi.size, psi.data, self._mode_prop)
		self._projector(psi.data)
		self._renormalize(psi, self._N_target)
		return self._p.dt

	def create(self, N, time=None, precision=None, random_init=True, E_modifier=0, psi=None):
		assert time is not None or precision is not None

		if isinstance(N, int):
			N = (N,)
		self._N_target = numpy.array(N)

		if psi is None:
			psi = WavefunctionSet(self._env, self._constants, self._grid, components=2)

			if random_init:
				psi.fillWithRandoms(1)
			else:
				psi.fillWithValue(1)

			self._renormalize(psi, self._N_target)

		self.prepare(components=2)

		if time is not None:
			callbacks = None
		elif precision is not None:
			callbacks = [
				EnergyCondition(self._env, self._constants, self._grid,
					precision=precision / self._p.dt,
					E_modifier=E_modifier)
			]
			time = 1e30

		if callbacks is not None:
			self.energy_data = callbacks[0].getData()

		self.run(psi, time, callbacks=callbacks)
		return psi


class SOGroundState(ImaginaryTimeGroundState):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		ImaginaryTimeGroundState.__init__(self, env, constants, grid)
		self._projector = Projector(env, constants, grid)
		self._addParameters(dt=1e-5, itmax=3, precision=1e-6,
			fix_total_N=True)
		self.prepare(**kwds)

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=1)

		self._p.relative_precision = self._p.precision / self._p.dt
		self._p.g_intra = self._constants.g_intra / self._constants.hbar
		self._p.g_inter = self._constants.g_inter / self._constants.hbar

		self._potentials = self._env.toDevice(
			getPotentials(self._constants, self._grid))
		self._mode_prop = self._env.toDevice(getSOEnergyExp(
			self._constants, self._grid, dt=self._p.dt / 2, imaginary_time=True))

	def _gpu__prepare_specific(self, **kwds):
		kernel_template = """
			// Propagates psi function in mode space
			EXPORTED_FUNC void mpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *mode_prop)
			{
				LIMITED_BY(gsize);

				COMPLEX mode_prop00 = mode_prop[GLOBAL_INDEX];
				COMPLEX mode_prop01 = mode_prop[GLOBAL_INDEX + ${g.msize}];
				COMPLEX mode_prop10 = mode_prop[GLOBAL_INDEX + ${g.msize * 2}];
				COMPLEX mode_prop11 = mode_prop[GLOBAL_INDEX + ${g.msize * 3}];

				COMPLEX data0 = data[GLOBAL_INDEX];
				COMPLEX data1 = data[GLOBAL_INDEX + ${g.msize}];

				data[GLOBAL_INDEX] = complex_mul(data0, mode_prop00) +
					complex_mul(data1, mode_prop01);
				data[GLOBAL_INDEX + ${g.msize}] = complex_mul(data0, mode_prop10) +
					complex_mul(data1, mode_prop11);
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

				<%
					p.gg = [
						[p.g_intra, p.g_inter],
						[p.g_inter, p.g_intra]
					]
				%>

				// iterate to midpoint solution
				%for i in range(p.itmax):
					// calculate midpoint log derivative and exponentiate
					%for comp in xrange(p.components):
					n${comp} = squared_abs(val${comp});
					%endfor

					%for comp in xrange(p.components):
					dval${comp} = exp((SCALAR)${p.dt / 2.0} * (-V
						%for other_comp in xrange(p.components):
						- (SCALAR)${p.gg[comp][other_comp]} * n${other_comp}
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
		data_copy = data.copy()
		data[0] = mode_prop[0, 0] * data_copy[0] + mode_prop[0, 1] * data_copy[1]
		data[1] = mode_prop[1, 0] * data_copy[0] + mode_prop[1, 1] * data_copy[1]

	def _cpu__kernel_xpropagate(self, gsize, data, potentials):
		data_copy = data.copy()

		g_intra = self._p.g_intra
		g_inter = self._p.g_inter

		dt = -self._p.dt / 2
		tile = (self._p.components, 1,) + (1,) * self._grid.dim
		p_tiled = numpy.tile(potentials, tile)

		for i in xrange(self._p.itmax):
			n = numpy.abs(data) ** 2
			dp = p_tiled.copy()

			dp[0] += n[0] * g_intra + n[1] * g_inter
			dp[1] += n[0] * g_inter + n[1] * g_intra

			d = numpy.exp(dp * dt)
			data.flat[:] = (data_copy * d).flat

		data *= d

	def _toEvolutionSpace(self, psi):
		psi.toMSpace()

	def _toMeasurementSpace(self, psi):
		psi.toXSpace()

	def _total_E(self, psi, N):
		return self._statstics.getSOEnergy(psi, N=N).sum()

	def _total_mu(self, psi, N):
		return self._statistics.getSOMu(psi, N=N).sum()

	def _propagate(self, psi):
		self._kernel_mpropagate(psi.size, psi.data, self._mode_prop)
		self._toMeasurementSpace(psi)
		self._kernel_xpropagate(psi.size, psi.data, self._potentials)
		self._toEvolutionSpace(psi)
		self._kernel_mpropagate(psi.size, psi.data, self._mode_prop)
		self._projector(psi.data)
		return self._p.dt


