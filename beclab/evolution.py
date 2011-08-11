"""
Classes, modeling the evolution of BEC.
"""

import numpy
import copy

from .helpers import *
from .constants import getPotentials, getPlaneWaveEnergy, getHarmonicEnergy, \
	getProjectorMask, UniformGrid, HarmonicGrid, WIGNER, CLASSICAL
from .ground_state import RK5Propagation, Projector


class TerminateEvolution(Exception):
	pass


class Evolution(PairedCalculation):

	def __init__(self, env):
		PairedCalculation.__init__(self, env)

	def _toMeasurementSpace(self, psi):
		pass

	def _toEvolutionSpace(self, psi):
		pass

	def _collectMetrics(self, t):
		pass

	def _runCallbacks(self, psi, callbacks):
		if callbacks is None:
			return

		self._toMeasurementSpace(psi)
		for callback in callbacks:
			callback(psi.time, psi)
		self._toEvolutionSpace(psi)

	def run(self, psi, time, callbacks=None, callback_dt=0):

		starting_time = psi.time
		callback_t = 0
		time_till_finish = time

		self._toEvolutionSpace(psi)

		try:
			self._runCallbacks(psi, callbacks)

			# 1e-10 modifier allow us to avoid cases when passed time
			# is very close to time (due to floating point errors)
			# which result in double calls to callbacks
			while time_till_finish > 1e-10:

				max_dt = (callback_dt - callback_t) if callback_dt != 0 else time_till_finish
				dt_used = self.propagate(psi, psi.time - starting_time, max_dt)
				self._collectMetrics(psi.time)

				psi.time += dt_used
				callback_t += dt_used
				time_till_finish -= dt_used

				if dt_used == max_dt:
					self._runCallbacks(psi, callbacks)
					callback_t = 0

			# if some time passed after the last execution of callbacks,
			# run them again
			if callback_t != 0:
				self._runCallbacks(psi, callbacks)

			self._toMeasurementSpace(psi)

		except TerminateEvolution:
			return psi.time


class NoisePropagator(PairedCalculation):

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self._random = createRandom(env, constants.double)

		self._addParameters(ensembles=1, components=2, order=1)
		self.prepare(**kwds)

	def _prepare(self):
		if isinstance(self._grid, UniformGrid):
			self._p.grid_size = self._grid.size
			self._p.comp_size = self._grid.size * self._p.ensembles
			grid_shape = self._grid.shape
			dV = self._grid.dV
		else:
			self._p.grid_size = self._grid.sizes[self._p.order]
			self._p.comp_size = self._p.grid_size * self._p.ensembles
			grid_shape = self._grid.shapes[self._p.order]
			dV = self._grid.dVs[self._p.order]

		self._normalization = self._env.toDevice(
			numpy.sqrt(1.0 / dV).astype(self._constants.scalar.dtype))

		self._p.losses_diffusion = copy.deepcopy(self._constants.losses_diffusion)

		self._randoms = self._env.allocate(
			(2, self._constants.noise_sources, self._p.ensembles) + grid_shape,
			dtype=self._constants.complex.dtype
		)

	def _gpu__prepare_specific(self):

		kernels = """
			<%!
				from math import sqrt, pi
			%>

			INTERNAL_FUNC void noise_func(
				COMPLEX G[${p.components}][${c.noise_sources}],
				COMPLEX vals[${p.components}])
			{
				<%
					def complex_mul_sequence(s):
						if len(s) == 1:
							return s[0]
						else:
							return 'complex_mul(' + s[0] + ', ' + complex_mul_sequence(s[1:]) + ')'
				%>

				%for comp in xrange(p.components):
				%for i, e in enumerate(p.losses_diffusion[comp]):
				<%
					coeff, orders = e
					sequence = []
					for j, order in enumerate(orders):
						if order > 0:
							sequence += ['vals[' + str(j) + ']'] * order
					if len(sequence) > 0:
						product = complex_mul_sequence(sequence)
					else:
						product = 'complex_ctr(1, 0)'
				%>
				%if coeff != 0:
				G[${comp}][${i}] = complex_mul_scalar(
					${product},
					(SCALAR)${coeff}
				);
				%else:
				G[${comp}][${i}] = complex_ctr(0, 0);
				%endif
				%endfor
				%endfor
			}

			EXPORTED_FUNC void add_noise(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *randoms, GLOBAL_MEM SCALAR *normalization, SCALAR dt)
			{
				LIMITED_BY(${p.comp_size});

				COMPLEX G[${p.components}][${c.noise_sources}];
				COMPLEX Z[2][${c.noise_sources}];
				COMPLEX vals0[${p.components}];
				COMPLEX vals[${p.components}];

				// load data
				%for comp in xrange(p.components):
				vals0[${comp}] = data[GLOBAL_INDEX + ${comp * p.comp_size}];
				vals[${comp}] = vals0[${comp}];
				%endfor

				SCALAR norm = normalization[GLOBAL_INDEX % ${p.grid_size}];

				// load randoms
				%for stage in xrange(2):
				%for ns in xrange(c.noise_sources):
				Z[${stage}][${ns}] = randoms[GLOBAL_INDEX +
					${stage * p.comp_size * c.noise_sources + ns * p.comp_size}];
				Z[${stage}][${ns}] = complex_mul_scalar(Z[${stage}][${ns}], norm);
				%endfor
				%endfor

				// first stage
				noise_func(G, vals);

				// second stage
				%for comp in xrange(p.components):
				vals[${comp}] = vals[${comp}] + complex_mul_scalar(
					%for ns in xrange(c.noise_sources):
					+ complex_mul(G[${comp}][${ns}], Z[0][${ns}])
					%endfor
					, sqrt(dt / 2));
				%endfor
				noise_func(G, vals);

				// write data to memory
				%for comp in xrange(p.components):
				vals0[${comp}] = vals0[${comp}] + complex_mul_scalar(
					%for ns in xrange(c.noise_sources):
					+ complex_mul(G[${comp}][${ns}], Z[1][${ns}])
					%endfor
					, sqrt(dt));
				data[GLOBAL_INDEX + ${comp * p.comp_size}] = vals0[${comp}];
				%endfor
			}
		"""

		self.__program = self.compileProgram(kernels)
		self._kernel_add_noise = self.__program.add_noise

	def _noiseFunc(self, data):

		G = numpy.empty((self._p.components, self._constants.noise_sources,) + data.shape[1:],
			dtype=data.dtype)

		for comp in xrange(self._p.components):
			for i, e in enumerate(self._p.losses_diffusion[comp]):
				coeff, orders = e
				if coeff != 0:
					res = coeff
					for j, order in enumerate(orders):
						if order > 0:
							res = res * data[j]
					G[comp][i] = res
				else:
					G[comp][i] = numpy.zeros(data.shape[1:], data.dtype)

		return G

	def _cpu__kernel_add_noise(self, gsize, data, randoms, normalization, dt):

		tile = (self._p.components,) + (1,) * (self._grid.dim + 2)

		normalization = numpy.tile(
			normalization,
			randoms.shape[1:3] + (1,) * self._grid.dim
		)

		G0 = self._noiseFunc(data)
		G1 = self._noiseFunc(data + numpy.sqrt(dt / 2) * (
			(G0 * numpy.tile(randoms[0] * normalization, tile)).sum(1)
		))

		data += (G1 * numpy.tile(randoms[1] * normalization, tile)).sum(1) * numpy.sqrt(dt)

	def propagateNoise(self, psi, dt):
		self._random.random_normal(self._randoms)
		self._kernel_add_noise(psi.size, psi.data, self._randoms,
			self._normalization, self._constants.scalar.cast(dt))


class SplitStepEvolution(Evolution):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)

		Evolution.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self._noise_prop = NoisePropagator(env, constants, grid)

		self._kdt = 0

		self._potentials = getPotentials(self._env, self._constants, self._grid)

		self._projector = Projector(self._env, self._constants, self._grid)

		self._addParameters(f_rabi=0, f_detuning=0, dt=1e-5, noise=False,
			ensembles=1, itmax=3, components=2)
		self.prepare(**kwds)

	def _prepare(self):
		# FIXME: matrix exponent in xpropagate() requires 2 components
		# different number will require significant changes
		assert self._p.components == 2

		self._projector.prepare(components=self._p.components, ensembles=self._p.ensembles)

		self._p.w_detuning = 2 * numpy.pi * self._p.f_detuning
		self._p.w_rabi = 2 * numpy.pi * self._p.f_rabi

		self._kvectors = getPlaneWaveEnergy(self._env, self._constants, self._grid)

		self._p.grid_size = self._grid.size
		self._p.comp_size = self._grid.size * self._p.ensembles

		self._p.g = self._constants.g / self._constants.hbar

		self._p.losses_drift = copy.deepcopy(self._constants.losses_drift)

		if self._p.noise:
			self._noise_prop.prepare(components=self._p.components, ensembles=self._p.ensembles)

	def _gpu__prepare_specific(self):

		kernels = """
			<%!
				from math import sqrt, pi
			%>

			// Propagates state vector in k-space
			EXPORTED_FUNC void kpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *kvectors, SCALAR dt)
			{
				LIMITED_BY(${p.comp_size});

				SCALAR k = kvectors[GLOBAL_INDEX % ${p.grid_size}];
				COMPLEX mode_coeff = cexp(1, -k * dt);
				COMPLEX val;

				%for comp in xrange(p.components):
					val = data[GLOBAL_INDEX + ${comp * p.comp_size}];
					data[GLOBAL_INDEX + ${comp * p.comp_size}] = complex_mul(
						val, mode_coeff
					);
				%endfor
			}

			// Propagates state vector in x-space
			EXPORTED_FUNC void xpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR t, SCALAR dt, SCALAR phi)
			{
				LIMITED_BY(${p.comp_size});

				SCALAR V = potentials[GLOBAL_INDEX % ${p.grid_size}];

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + ${comp * p.comp_size}];
				COMPLEX val0_${comp} = val${comp}; // initial field value
				SCALAR n${comp};
				COMPLEX N${comp};
				%endfor

				%if p.f_rabi == 0:
				%for comp in xrange(p.components):
					COMPLEX dval${comp};
				%endfor
				%else:
					SCALAR k, f;
					COMPLEX rt, l0, l1, ev10, ev11, ev_inv_coeff, l0_exp, l1_exp;
					COMPLEX m[4];
				%endif

				// iterate to midpoint solution
				%for iter in range(p.itmax):
					%for comp in xrange(p.components):
					n${comp} = squared_abs(val${comp});
					%endfor

					%for comp in xrange(p.components):
					N${comp} = complex_ctr(
						%if len(p.losses_drift[comp]) > 0:
						%for coeff, orders in p.losses_drift[comp]:
						-(SCALAR)${coeff}
							%for loss_comp, order in enumerate(orders):
							${(' * ' + ' * '.join(['n' + str(loss_comp)] * order)) if order != 0 else ''}
							%endfor
						%endfor
						%else:
						0
						%endif
						,
						-V
						%for comp_other in xrange(p.components):
						-(SCALAR)${p.g[comp, comp_other]} * n${comp_other}
						%endfor
					);
					%endfor

					%if p.f_rabi == 0:
						%for comp in xrange(p.components):
						// calculate midpoint log derivative and exponentiate
						dval${comp} = cexp(complex_mul_scalar(N${comp}, ${p.dt / 2}));

						// propagate to midpoint using log derivative
						val${comp} = complex_mul(val0_${comp}, dval${comp});
						%endfor
					%else:

						k = (SCALAR)${p.w_rabi};
						f = (SCALAR)${p.w_detuning} * t + phi;

						// calculating exp([[N1, -ik exp(-if)/2], [-ik exp(if)/2, N2]])
						rt = csqrt(
							complex_ctr(-k * k, 0) +
							complex_mul(N0 - N1, N0 - N1));

						l0 = complex_mul_scalar(N0 + N1 - rt, 0.5); // eigenvalues 1
						l1 = complex_mul_scalar(rt + N0 + N1, 0.5); // eigenvalues 2

						// elements of eigenvector matrix ([1, 1], [ev10, ev11])
						ev10 = complex_mul(
							cexp((SCALAR)-1.0 / k, f + (SCALAR)${pi / 2}),
							rt + N0 - N1);
						ev11 = complex_mul(
							cexp((SCALAR)1.0 / k, f + (SCALAR)${pi / 2}),
							rt - N0 + N1);

						// elements of inverse eigenvector matrix
						// ([-ev11, 1], [ev10, -1]) * ev_inf_coeff
						ev_inv_coeff = complex_div(complex_ctr(1, 0), ev10 - ev11);

						l0_exp = cexp(complex_mul_scalar(l0, ${p.dt / 2}));
						l1_exp = cexp(complex_mul_scalar(l1, ${p.dt / 2}));

						m[0] = complex_mul(
							complex_mul(l1_exp, ev10) - complex_mul(l0_exp, ev11),
							ev_inv_coeff);
						m[1] = complex_mul(l0_exp - l1_exp, ev_inv_coeff);
						m[2] = complex_mul(
							complex_mul(complex_mul(ev10, ev11), l1_exp - l0_exp),
							ev_inv_coeff);
						m[3] = complex_mul(
							complex_mul(l0_exp, ev10) - complex_mul(l1_exp, ev11),
							ev_inv_coeff);

						// propagate to midpoint
						val0 = complex_mul(val0_0, m[0]) + complex_mul(val0_1, m[1]);
						val1 = complex_mul(val0_0, m[2]) + complex_mul(val0_1, m[3]);

					%endif

				%endfor

				// propagate to endpoint using log derivative
				%if p.f_rabi == 0:
					%for comp in xrange(p.components):
					data[GLOBAL_INDEX + ${comp * p.comp_size}] =
						complex_mul(val${comp}, dval${comp});
					%endfor
				%else:
					data[GLOBAL_INDEX + ${0 * p.comp_size}] =
						complex_mul(val0, m[0]) + complex_mul(val1, m[1]);
					data[GLOBAL_INDEX + ${1 * p.comp_size}] =
						complex_mul(val0, m[2]) + complex_mul(val1, m[3]);
				%endif
			}
		"""

		self.__program = self.compileProgram(kernels)
		self._kernel_kpropagate = self.__program.kpropagate
		self._kernel_xpropagate = self.__program.xpropagate

	def _cpu__kernel_kpropagate(self, gsize, data, kvectors, dt):
		data *= numpy.tile(numpy.exp(kvectors * (-1j * dt)),
			(self._p.components, self._p.ensembles) + (1,) * self._grid.dim)

	def _cpu__kernel_xpropagate(self, gsize, data, potentials, t, dt, phi):
		data0 = data.copy()
		g = self._p.g

		l = self._p.losses_drift
		V = numpy.tile(self._potentials * 1j,
			(self._p.ensembles,) + (1,) * self._grid.dim)

		m = numpy.empty((2, 2) + data.shape[1:], dtype=self._constants.complex.dtype)
		N = numpy.empty_like(data)

		for iter in xrange(self._p.itmax):
			n = numpy.abs(data) ** 2

			for comp in xrange(self._p.components):
				N[comp].flat[:] = (-V).flat
				for comp_other in xrange(self._p.components):
					N[comp] -= 1j * n[comp_other] * g[comp, comp_other]

				for coeff, orders in l[comp]:
					to_add = -coeff
					for i, order in enumerate(orders):
						to_add = to_add * (n[i] ** order)
					N[comp] += to_add

			if self._p.w_rabi == 0:

				ddata = numpy.exp(N * (dt / 2))
				data.flat[:] = (data0 * ddata).flat

			else:

				k = self._p.w_rabi
				f = self._p.w_detuning * t + phi

				# calculating exp([[N1, -ik exp(-if)/2], [-ik exp(if)/2, N2]])
				rt = numpy.sqrt(-k ** 2 + (N[0] - N[1]) ** 2 + 0j)

				l0 = 0.5 * (-rt + N[0] + N[1]) # eigenvalues 1
				l1 = 0.5 * (rt + N[0] + N[1]) # eigenvalues 2

				#ev = numpy.array([
				#	[1, -(1j * numpy.exp(1j * f) * (rt + N1 - N2)) / k],
				#	[1, (1j * numpy.exp(1j * f) * (rt - N1 + N2)) / k]
				#])

				# elements of eigenvector matrix ([1, 1], [ev10, ev11])
				ev10 = -(1j * numpy.exp(1j * f) * (rt + N[0] - N[1])) / k
				ev11 = (1j * numpy.exp(1j * f) * (rt - N[0] + N[1])) / k

				# elements of inverse eigenvector matrix
				# ([-ev11, 1], [ev10, -1]) / ev_inf_coeff
				ev_inv_coeff = ev10 - ev11

				l0_exp = numpy.exp(l0 * dt / 2)
				l1_exp = numpy.exp(l1 * dt / 2)

				m[0,0].flat[:] = ((l1_exp * ev10 - l0_exp * ev11) / ev_inv_coeff).flat
				m[0,1].flat[:] = ((l0_exp - l1_exp) / ev_inv_coeff).flat
				m[1,0].flat[:] = ((ev10 * ev11 * (l1_exp - l0_exp)) / ev_inv_coeff).flat
				m[1,1].flat[:] = ((l0_exp * ev10 - l1_exp * ev11) / ev_inv_coeff).flat

				data[0] = m[0,0] * data0[0] + m[0,1] * data0[1]
				data[1] = m[1,0] * data0[0] + m[1,1] * data0[1]

		if self._p.w_rabi == 0:
			data *= ddata
		else:
			data_copy = data.copy()
			data[0] = m[0,0] * data_copy[0] + m[0,1] * data_copy[1]
			data[1] = m[1,0] * data_copy[0] + m[1,1] * data_copy[1]

	def _kpropagate(self, psi, dt):
		self._kdt += dt

	def _xpropagate(self, psi, t, phi, dt):
		cast = self._constants.scalar.cast
		self._kernel_xpropagate(psi.size, psi.data, self._potentials,
		cast(t), cast(dt), cast(phi))

	def _finish_kpropagate(self, psi):
		if self._kdt != 0:
			self._kernel_kpropagate(psi.size, psi.data,
				self._kvectors, self._constants.scalar.cast(self._kdt))
			self._kdt = 0

	def _toMeasurementSpace(self, psi):
		self._finish_kpropagate(psi)
		psi.toXSpace()

	def _toEvolutionSpace(self, psi):
		psi.toMSpace()

	def propagate(self, psi, t, max_dt):

		dt = self._p.dt
		if max_dt < dt:
			dt = max_dt

		self._kpropagate(psi, dt / 2)
		self._finish_kpropagate(psi)

		psi.toXSpace()
		self._xpropagate(psi, t, self._phi, dt)

		if self._p.noise:
			self._noise_prop.propagateNoise(psi, dt)

		self._midstep = True
		psi.toMSpace()
		self._projector(psi.data)

		self._kpropagate(psi, dt / 2)

		return dt

	def run(self, psi, *args, **kwds):

		if 'callbacks' in kwds:
			for cb in kwds['callbacks']:
				cb.prepare(components=psi.components, ensembles=psi.ensembles)

		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		self.prepare(ensembles=psi.ensembles, components=psi.components,
			noise=(psi.type == WIGNER))
		Evolution.run(self, psi, *args, **kwds)


class RK5IPEvolution(Evolution):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)
		Evolution.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self._plan = createFFTPlan(self._env, self._constants, self._grid)
		self._potentials = getPotentials(self._env, self._constants, self._grid)
		self._energy = getPlaneWaveEnergy(self._env, self._constants, self._grid)

		self._projector = Projector(self._env, self._constants, self._grid)

		self._propagator = RK5Propagation(self._env, self._constants, self._grid, mspace=False)
		self._noise_prop = NoisePropagator(self._env, self._constants, self._grid)

		self._addParameters(atol_coeff=1e-3, eps=1e-6, dt_guess=1e-4, Nscale=10000,
			components=2, ensembles=1, f_detuning=0, f_rabi=0, noise=False)
		self.prepare(**kwds)

	def _prepare(self):
		# FIXME: coupling terms assume that there are two components
		assert self._p.f_rabi == 0 or self._p.components == 2

		self._projector.prepare(components=self._p.components, ensembles=self._p.ensembles)

		self._p.w_detuning = 2 * numpy.pi * self._p.f_detuning
		self._p.w_rabi = 2 * numpy.pi * self._p.f_rabi

		self._p.comp_size = self._grid.size * self._p.ensembles
		self._p.grid_size = self._grid.size
		self._p.losses_drift = copy.deepcopy(self._constants.losses_drift)

		# FIXME: toIP() can be done inplace in self._k,
		# we just need to support offsets in FFT (because I only want to transform part of it)
		self._buffer = self._env.allocate(
			(self._p.components, self._p.ensembles) + self._grid.shape,
			self._constants.complex.dtype
		)

		if self._p.noise:
			self._noise_prop.prepare(components=self._p.components, ensembles=self._p.ensembles)

		self._p.g = self._constants.g / self._constants.hbar

		mu = self._constants.muTF(self._p.Nscale, dim=self._grid.dim, comp=0)
		peak_density = numpy.sqrt(mu / self._constants.g[0, 0])

		self._propagator.prepare(eps=self._p.eps, dt_guess=self._p.dt_guess, mspace=False,
			tiny=peak_density * self._p.atol_coeff, components=self._p.components, ensembles=self._p.ensembles)

	def _gpu__prepare_specific(self):
		kernel_template = """
			<%
				from math import pi
			%>

			EXPORTED_FUNC void transformIP(
				int gsize, GLOBAL_MEM COMPLEX *data, GLOBAL_MEM SCALAR *energy, SCALAR dt)
			{
				LIMITED_BY(${p.comp_size});
				COMPLEX val;
				int id;
				SCALAR e = energy[GLOBAL_INDEX % ${p.grid_size}];

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + ${comp * p.comp_size};
				val = data[id];
				data[id] = complex_mul(val, cexp(1, e * dt));
				%endfor
			}

			EXPORTED_FUNC void propagationFunc(int gsize,
				GLOBAL_MEM COMPLEX *result, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0, SCALAR t, SCALAR phi)
			{
				LIMITED_BY(${p.comp_size});
				SCALAR V = potentials[GLOBAL_INDEX % ${p.grid_size}];

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + ${comp * p.comp_size}];
				SCALAR n${comp} = squared_abs(val${comp});
				COMPLEX N${comp};
				%endfor

				%for comp in xrange(p.components):
				N${comp} = complex_mul(val${comp},
					complex_ctr(
					%if len(p.losses_drift[comp]) > 0:
					%for coeff, orders in p.losses_drift[comp]:
					-(SCALAR)${coeff}
						%for loss_comp, order in enumerate(orders):
						${(' * ' + ' * '.join(['n' + str(loss_comp)] * order)) if order != 0 else ''}
						%endfor
					%endfor
					%else:
					0
					%endif
					,
					-V
					%for comp_other in xrange(p.components):
					-(SCALAR)${p.g[comp, comp_other]} * n${comp_other}
					%endfor
					)
				);
				%endfor

				%if p.f_rabi != 0:
				SCALAR amplitude = ${p.w_rabi / 2};
				SCALAR phase = ${p.w_detuning} * t + phi;

				%for comp in xrange(p.components):
				COMPLEX coupling${comp} = cexp(
					amplitude,
					${'-' if comp == 0 else ''}phase - (SCALAR)${pi / 2}
				);
				%endfor

				%for comp in xrange(p.components):
				N${comp} = N${comp} + complex_mul(val${1 - comp}, coupling${comp});
				%endfor
				%endif

				%for comp in xrange(p.components):
				result[GLOBAL_INDEX + ${p.comp_size * comp}] =
					complex_mul_scalar(N${comp}, dt0);
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_transformIP = self._program.transformIP
		self._kernel_propagationFunc = self._program.propagationFunc

	def _cpu__kernel_transformIP(self, gsize, data, energy, dt):
		data *= numpy.tile(numpy.exp(energy * (1j * dt)),
			(self._p.components, self._p.ensembles,) + (1,) * self._grid.dim)

	def _cpu__kernel_propagationFunc(self, gsize, result, data, potentials, dt0, t, phi):
		g = self._p.g
		l = self._p.losses_drift
		n = numpy.abs(data) ** 2

		tile = (self._p.components, self._p.ensembles) + (1,) * self._grid.dim
		p_tiled = numpy.tile(-1j * potentials, tile)

		result.flat[:] = p_tiled.flat

		for comp in xrange(self._p.components):
			for comp_other in xrange(self._p.components):
				result[comp] -= 1j * n[comp_other] * g[comp, comp_other]

			for coeff, orders in l[comp]:
				to_add = -coeff
				for i, order in enumerate(orders):
					to_add = to_add * (n[i] ** order)
				result[comp] += to_add

		for comp in xrange(self._p.components):
			result[comp] *= data[comp]

		if self._p.f_rabi != 0:
			amplitude = self._p.w_rabi / 2
			phase = self._p.w_detuning * t + phi

			result[0] += data[1] * amplitude * numpy.exp(-1j * (phase + numpy.pi / 2))
			result[1] += data[0] * amplitude * numpy.exp(1j * (phase - numpy.pi / 2))

		result *= dt0

	def _propFunc(self, results, values, dt, dt_full, stage):
		self._fromIP(values, dt, False)
		cast = self._constants.scalar.cast
		self._kernel_propagationFunc(self._p.comp_size, self._buffer, values,
			self._potentials, cast(dt_full), cast(self._t), cast(self._phi))
		self._toIP(self._buffer, dt, True)
		self._env.copyBuffer(self._buffer, dest=results,
			dest_offset=stage * self._p.comp_size * self._p.components)

	def _finalizeFunc(self, psi, dt_used):
		self._fromIP(psi.data, dt_used, False)

	def propagate(self, psi, t, max_dt):
		self._t = t
		dt_used = self._propagator.propagate(self._propFunc, self._finalizeFunc, psi,
			max_dt=max_dt)
		if self._p.noise:
			self._noise_prop.propagateNoise(psi, dt)
			if not self._projector.is_identity:
				batch = self._p.components * self._p.ensembles
				self._plan.execute(psi.data, batch=batch)
				self._projector(psi.data)
				self._plan.execute(psi.data, batch=batch, inverse=True)

		return dt_used

	def _toIP(self, data, dt, project):
		batch = self._p.components * self._p.ensembles
		self._plan.execute(data, batch=batch)
		if dt != 0.0:
			self._kernel_transformIP(self._p.comp_size,
				data, self._energy, self._constants.scalar.cast(dt))
		if project:
			self._projector(data)
		self._plan.execute(data, batch=batch, inverse=True)

	def _fromIP(self, data, dt, project):
		self._toIP(data, -dt, project)

	def run(self, psi, *args, **kwds):

		if 'callbacks' in kwds:
			for cb in kwds['callbacks']:
				cb.prepare(components=psi.components, ensembles=psi.ensembles)

		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		self.prepare(ensembles=psi.ensembles, components=psi.components,
			noise=(psi.type == WIGNER))
		Evolution.run(self, psi, *args, **kwds)


class RK5HarmonicEvolution(Evolution):

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, HarmonicGrid)
		Evolution.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self._energy = getHarmonicEnergy(self._env, self._constants, self._grid)
		self._plan3 = createFHTPlan(env, constants, grid, 3)

		self._projector = Projector(env, constants, grid)
		self._propagator = RK5Propagation(self._env, self._constants, self._grid, mspace=True)
		self._noise_prop = NoisePropagator(self._env, self._constants, self._grid)

		self._addParameters(kwds, atol_coeff=1e-3, eps=1e-6,
			dt_guess=1e-4, Nscale=10000, components=2, ensembles=1,
			f_detuning=0, f_rabi=0, noise=False)

	def _prepare(self):
		self._projector.prepare(components=self._p.components, ensembles=self._p.ensembles)
		self._p.g = self._constants.g / self._constants.hbar

		shape = self._grid.mshape
		cdtype = self._constants.complex.dtype

		self._p.w_detuning = 2 * numpy.pi * self._p.f_detuning
		self._p.w_rabi = 2 * numpy.pi * self._p.f_rabi

		self._x3data = self._env.allocate((self._p.components, self._p.ensembles) + self._grid.shapes[3], dtype=cdtype)
		self._p.comp_size3 = self._grid.sizes[3] * self._p.ensembles
		self._p.comp_msize = self._grid.msize * self._p.ensembles
		self._p.grid_msize = self._grid.msize

		self._mdata = self._env.allocate((self._p.components, self._p.ensembles) + self._grid.mshape, dtype=cdtype)

		self._p.losses_drift = copy.deepcopy(self._constants.losses_drift)

		self._propagator.prepare(eps=self._p.eps, dt_guess=self._p.dt_guess,
			tiny=numpy.sqrt(self._p.Nscale) * self._p.atol_coeff, components=self._p.components,
			ensembles=self._p.ensembles)

		if self._p.noise:
			self._noise_prop.prepare(components=self._p.components, ensembles=self._p.ensembles)

	def _gpu__prepare_specific(self):
		kernel_template = """
			<%
				from math import pi
			%>

			EXPORTED_FUNC void calculateNonlinear(int gsize, GLOBAL_MEM COMPLEX *data,
				SCALAR t, SCALAR phi)
			{
				LIMITED_BY(${p.comp_size3});

				%for comp in xrange(p.components):
				COMPLEX val${comp} = data[GLOBAL_INDEX + ${comp * p.comp_size3}];
				SCALAR n${comp} = squared_abs(val${comp});
				COMPLEX N${comp};
				%endfor

				%for comp in xrange(p.components):
				N${comp} = complex_mul(val${comp},
					complex_ctr(
					%if len(p.losses_drift[comp]) > 0:
					%for coeff, orders in p.losses_drift[comp]:
					-(SCALAR)${coeff}
						%for loss_comp, order in enumerate(orders):
						${(' * ' + ' * '.join(['n' + str(loss_comp)] * order)) if order != 0 else ''}
						%endfor
					%endfor
					%else:
					0
					%endif
					,
					%for comp_other in xrange(p.components):
					-(SCALAR)${p.g[comp, comp_other]} * n${comp_other}
					%endfor
					)
				);
				%endfor

				%if p.f_rabi != 0:
				SCALAR amplitude = ${p.w_rabi / 2};
				SCALAR phase = ${p.w_detuning} * t + phi;

				%for comp in xrange(p.components):
				COMPLEX coupling${comp} = cexp(
					amplitude,
					${'-' if comp == 0 else ''}phase - (SCALAR)${pi / 2}
				);
				%endfor

				%for comp in xrange(p.components):
				N${comp} = N${comp} + complex_mul(val${1 - comp}, coupling${comp});
				%endfor
				%endif

				%for comp in xrange(p.components):
				data[GLOBAL_INDEX + ${p.comp_size3 * comp}] = N${comp};
				%endfor
			}

			EXPORTED_FUNC void propagationFunc(int gsize,
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *nldata, GLOBAL_MEM SCALAR *energy, SCALAR dt0, int stage)
			{
				LIMITED_BY(${p.comp_msize});
				COMPLEX val, nlval;
				SCALAR e = energy[GLOBAL_INDEX % ${p.grid_msize}];

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + ${p.comp_msize * comp}];
				nlval = nldata[GLOBAL_INDEX + ${p.comp_msize * comp}];
				k[GLOBAL_INDEX + ${p.comp_msize * p.components} * stage + ${p.comp_msize * comp}] =
					complex_mul_scalar(
						complex_mul(val, complex_ctr(0, -e)) + nlval,
						dt0);
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_propagationFunc = self._program.propagationFunc
		self._kernel_calculateNonlinear = self._program.calculateNonlinear

	def _cpu__kernel_calculateNonlinear(self, gsize, data, t, phi):
		g = self._p.g
		l = self._p.losses_drift
		n = numpy.abs(data) ** 2
		data_copy = data.copy()

		for comp in xrange(self._p.components):
			data[comp] = 0
			for comp_other in xrange(self._p.components):
				data[comp] -= 1j * n[comp_other] * g[comp, comp_other]

			for coeff, orders in l[comp]:
				to_add = -coeff
				for i, order in enumerate(orders):
					to_add = to_add * (n[i] ** order)
				data[comp] += to_add

		for comp in xrange(self._p.components):
			data[comp] *= data_copy[comp]

		if self._p.f_rabi != 0:
			amplitude = self._p.w_rabi / 2
			phase = self._p.w_detuning * t + phi

			data[0] += data_copy[1] * amplitude * numpy.exp(-1j * (phase + numpy.pi / 2))
			data[1] += data_copy[0] * amplitude * numpy.exp(1j * (phase - numpy.pi / 2))

	def _cpu__kernel_propagationFunc(self, gsize, k, data, nldata, energy, dt0, stage):
		tile = (self._p.components,) + (1,) * (self._grid.dim + 1)
		e = numpy.tile(-1j * energy, tile)
		k[stage].flat[:] = ((data * e + nldata) * dt0).flat

	def _propFunc(self, results, values, dt, dt_full, stage):
		cast = self._constants.scalar.cast
		batch = self._p.components * self._p.ensembles
		self._plan3.execute(values, self._x3data, inverse=True, batch=batch)
		self._kernel_calculateNonlinear(self._p.comp_size3, self._x3data,
			cast(self._t), cast(self._phi))
		self._plan3.execute(self._x3data, self._mdata, batch=batch)
		self._projector(self._mdata)
		self._kernel_propagationFunc(self._p.comp_msize, results, values,
			self._mdata, self._energy, cast(dt_full), numpy.int32(stage))

	def _toMeasurementSpace(self, psi):
		psi.toXSpace()

	def _toEvolutionSpace(self, psi):
		psi.toMSpace()

	def _finalizeFunc(self, psi, dt_used):
		pass

	def propagate(self, psi, t, max_dt):
		self._t = t
		dt_used = self._propagator.propagate(self._propFunc, self._finalizeFunc, psi,
			max_dt=max_dt)
		if self._p.noise:
			psi.toXSpace()
			self._noise_prop.propagateNoise(psi, dt_used)
			psi.toMSpace()
			if not self._projector.is_identity:
				self._projector(psi.data)

		return dt_used

	def run(self, psi, *args, **kwds):

		if 'callbacks' in kwds:
			for cb in kwds['callbacks']:
				cb.prepare(components=psi.components, ensembles=psi.ensembles)

		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		self.prepare(ensembles=psi.ensembles, components=psi.components,
			noise=(psi.type == WIGNER))
		Evolution.run(self, psi, *args, **kwds)
