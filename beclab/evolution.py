"""
Classes, modeling the evolution of BEC.
"""

import numpy
import copy

from .helpers import *
from .constants import getPotentials, getPlaneWaveEnergy, getHarmonicEnergy, \
	getProjectorMask, UniformGrid, HarmonicGrid, WIGNER, CLASSICAL
from .ground_state import RK5Propagation


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

	def run(self, psi, time=1.0, callbacks=None, callback_dt=0):

		starting_time = psi.time
		ending_time = psi.time + time
		callback_t = 0

		self._toEvolutionSpace(psi)

		try:
			self._runCallbacks(psi, callbacks)

			while psi.time - starting_time < time:
				dt_used = self.propagate(psi, psi.time - starting_time, callback_dt - callback_t)
				self._collectMetrics(psi.time)

				psi.time += dt_used
				callback_t += dt_used

				if callback_t >= callback_dt:
					self._runCallbacks(psi, callbacks)
					callback_t = 0

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

		self._addParameters(ensembles=1, components=2)
		self.prepare(**kwds)

	def _prepare(self):
		self._p.grid_size = self._grid.size
		self._p.comp_size = self._grid.size * self._p.ensembles

		self._p.losses_diffusion = copy.deepcopy(self._constants.losses_diffusion)

		self._randoms = self._env.allocate(
			(2, self._constants.noise_sources, self._p.ensembles) + self._grid.shape,
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
					normalization = sqrt(g.size / g.V)

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
					(SCALAR)${coeff * normalization}
				);
				%else:
				G[${comp}][${i}] = complex_ctr(0, 0);
				%endif
				%endfor
				%endfor
			}

			EXPORTED_FUNC void add_noise(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *randoms, SCALAR dt)
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

				// load randoms
				%for stage in xrange(2):
				%for ns in xrange(c.noise_sources):
				Z[${stage}][${ns}] = randoms[GLOBAL_INDEX +
					${stage * p.comp_size * c.noise_sources + ns * p.comp_size}];
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
		normalization = numpy.sqrt(self._grid.size / self._grid.V)

		G = numpy.empty((self._p.components, self._constants.noise_sources,) + data.shape[1:],
			dtype=data.dtype)

		for comp in xrange(self._p.components):
			for i, e in enumerate(self._p.losses_diffusion[comp]):
				coeff, orders = e
				if coeff != 0:
					res = coeff * normalization
					for j, order in enumerate(orders):
						if order > 0:
							res = res * data[j]
					G[comp][i] = res
				else:
					G[comp][i] = numpy.zeros(data.shape[1:], data.dtype)

		return G

	def _cpu__kernel_add_noise(self, gsize, data, randoms, dt):

		tile = (self._p.components,) + (1,) * (self._grid.dim + 2)

		G0 = self._noiseFunc(data)
		G1 = self._noiseFunc(data + numpy.sqrt(dt / 2) * (
			(G0 * numpy.tile(randoms, tile)).sum(1)
		))

		data += (G1 * numpy.tile(randoms, tile)).sum(1) * numpy.sqrt(dt)

	def propagateNoise(self, psi, dt):
		self._random.random_normal(self._randoms)
		self._kernel_add_noise(psi.size, psi.data, self._randoms, self._constants.scalar.cast(dt))


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

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

		self._potentials = getPotentials(self._env, self._constants, self._grid)

		self._projector_mask = getProjectorMask(self._env, self._constants, self._grid)

		self._addParameters(f_rabi=0, f_detuning=0, dt=1e-5, noise=False,
			ensembles=1, itmax=3, components=2)
		self.prepare(**kwds)

	def _prepare(self):
		# FIXME: matrix exponent in xpropagate() requires 2 components
		# different number will require significant changes
		assert self._p.components == 2

		self._p.w_detuning = 2 * numpy.pi * self._p.f_detuning
		self._p.w_rabi = 2 * numpy.pi * self._p.f_rabi

		kvectors = getPlaneWaveEnergy(None, self._constants, self._grid)
		self._mode_prop = self._env.toDevice(numpy.exp(kvectors * (-1j * self._p.dt / 2)))
		self._mode_prop2 = self._env.toDevice(numpy.exp(kvectors * (-1j * self._p.dt)))

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
				GLOBAL_MEM COMPLEX *mode_prop)
			{
				LIMITED_BY(${p.comp_size});

				COMPLEX mode_coeff = mode_prop[GLOBAL_INDEX % ${p.grid_size}];
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
				GLOBAL_MEM SCALAR *potentials, SCALAR t, SCALAR phi)
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

			EXPORTED_FUNC void projector(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *projector_mask)
			{
				LIMITED_BY(${p.comp_size});
				SCALAR mask_val = projector_mask[GLOBAL_INDEX % ${g.size}];
				COMPLEX val;

				%for comp in xrange(p.components):
				val = data[GLOBAL_INDEX + ${comp * p.comp_size}];
				data[GLOBAL_INDEX + ${comp * p.comp_size}] = complex_mul_scalar(val, mask_val);
				%endfor
			}
		"""

		self.__program = self.compileProgram(kernels)
		self._kernel_kpropagate = self.__program.kpropagate
		self._kernel_xpropagate = self.__program.xpropagate
		self._kernel_projector = self.__program.projector

	def _cpu__kernel_projector(self, gsize, data, projector_mask):
		mask = numpy.tile(projector_mask,
			(self._p.components, self._p.ensembles) + (1,) * self._grid.dim)

		data *= mask

	def _cpu__kernel_kpropagate(self, gsize, data, mode_prop):
		data *= numpy.tile(mode_prop,
			(self._p.components, self._p.ensembles) + (1,) * self._grid.dim)

	def _cpu__kernel_xpropagate(self, gsize, data, potentials, t, phi):
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

				ddata = numpy.exp(N * (self._p.dt / 2))
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

				l0_exp = numpy.exp(l0 * self._p.dt / 2)
				l1_exp = numpy.exp(l1 * self._p.dt / 2)

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

	def _projector(self, psi):
		self._kernel_projector(psi.size, psi.data, self._projector_mask)

	def _kpropagate(self, psi, half_step):
		self._kernel_kpropagate(psi.size, psi.data,
			self._mode_prop if half_step else self._mode_prop2)

	def _xpropagate(self, psi, t, phi):
		cast = self._constants.scalar.cast
		self._kernel_xpropagate(psi.size, psi.data, self._potentials, cast(t), cast(phi))

	def _finishStep(self, psi):
		if self._midstep:
			self._kpropagate(psi, False)
			self._midstep = False

	def _toMeasurementSpace(self, psi):
		self._finishStep(psi)
		psi.toXSpace()

	def _toEvolutionSpace(self, psi):
		psi.toMSpace()

	def propagate(self, psi, t, remaining_time):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them
		if self._midstep:
			self._kpropagate(psi, False)
		else:
			self._kpropagate(psi, True)

		if self._p.noise:
			self._projector(psi)

		psi.toXSpace()
		self._xpropagate(psi, t, self._phi)

		if self._p.noise:
			self._noise_prop.propagateNoise(psi, self._p.dt)

		self._midstep = True
		psi.toMSpace()

		return self._p.dt

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

		self._propagator = RK5Propagation(self._env, self._constants, self._grid, mspace=False)
		self._noise_prop = NoisePropagator(self._env, self._constants, self._grid)

		self._addParameters(atol_coeff=1e-3, eps=1e-6, dt_guess=1e-6, Nscale=10000,
			components=2, ensembles=1, f_detuning=0, f_rabi=0, noise=False)
		self.prepare(**kwds)

	def _prepare(self):
		# FIXME: coupling terms assume that there are two components
		assert self._p.f_rabi == 0 or self._p.components == 2

		self._p.w_detuning = 2 * numpy.pi * self._p.f_detuning
		self._p.w_rabi = 2 * numpy.pi * self._p.f_rabi

		self._p.comp_size = self._grid.size * self._p.ensembles
		self._p.grid_size = self._grid.size
		self._p.losses_drift = copy.deepcopy(self._constants.losses_drift)

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

			EXPORTED_FUNC void transformIP(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *energy, SCALAR dt)
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
				GLOBAL_MEM COMPLEX *k, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *potentials, SCALAR dt0, SCALAR t, SCALAR phi, int stage)
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
				)
				%endfor

				%for comp in xrange(p.components):
				N${comp} = N${comp} + complex_mul(val${1 - comp}, coupling${comp});
				%endfor
				%endif

##				%for comp in xrange(p.components):
##				k[GLOBAL_INDEX + ${p.comp_size * p.components} * stage + ${p.comp_size * comp}] =
##					complex_mul_scalar(N${comp}, dt0);
##				%endfor
				%for comp in xrange(p.components):
				k[GLOBAL_INDEX + ${p.comp_size * comp}] =
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

	def _cpu__kernel_propagationFunc(self, gsize, result, data, potentials, dt0, t, phi, stage):
		g = self._p.g
		l = self._p.losses_drift
		n = numpy.abs(data) ** 2

#		result = k[stage]

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
		self._fromIP(values, dt)
		cast = self._constants.scalar.cast
#		self._kernel_propagationFunc(self._p.comp_size, results, values,
#			self._potentials, cast(dt_full), cast(self._t), cast(self._phi), numpy.int32(stage))
		self._kernel_propagationFunc(self._p.comp_size, self._buffer, values,
			self._potentials, cast(dt_full), cast(self._t), cast(self._phi), numpy.int32(stage))
		self._toIP(self._buffer, dt)
		self._env.copyBuffer(self._buffer, dest=results,
			dest_offset=stage * self._p.comp_size * self._p.components)

	def _finalizeFunc(self, psi, dt_used):
		self._fromIP(psi.data, dt_used)

	def propagate(self, psi, t, remaining_time):
		self._t = t
		return self._propagator.propagate(self._propFunc, self._finalizeFunc, psi,
			max_dt=remaining_time)

	def _toIP(self, data, dt):
		if dt == 0.0:
			return

		batch = self._p.components * self._p.ensembles
		self._plan.execute(data, batch=batch)
		self._kernel_transformIP(self._p.comp_size,
			data, self._energy, self._constants.scalar.cast(dt))
		self._plan.execute(data, batch=batch, inverse=True)

	def _fromIP(self, data, dt):
		self._toIP(data, -dt)

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
