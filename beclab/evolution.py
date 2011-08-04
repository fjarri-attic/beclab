"""
Classes, modeling the evolution of BEC.
"""

import numpy
import copy

from .helpers import *
from .constants import getPotentials, getPlaneWaveEnergy, getHarmonicEnergy, \
	getProjectorMask, UniformGrid, HarmonicGrid, WIGNER, CLASSICAL


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










class RK4Evolution(Evolution):

	def __init__(self, env, constants, rabi_freq=0, detuning=0, dt=None):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		# FIXME: temporary stub; remove when implement constants/grid separation
		self._dt = dt if dt is not None else self._constants.dt_evo

		self._detuning = 2 * math.pi * detuning
		self._rabi_freq = 2 * math.pi * rabi_freq

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
		self._random = createRandom(env, constants.double)

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernels = """
			<%!
				from math import sqrt
			%>

			INTERNAL_FUNC void propagationFunc(COMPLEX *a_res, COMPLEX *b_res,
				COMPLEX a, COMPLEX b,
				COMPLEX ka, COMPLEX kb,
				SCALAR t, SCALAR dt,
				SCALAR kvector, SCALAR potential,
				SCALAR phi)
			{
				SCALAR n_a = squared_abs(a);
				SCALAR n_b = squared_abs(b);

				COMPLEX ta = complex_mul_scalar(ka, kvector) + complex_mul_scalar(a, potential);
				COMPLEX tb = complex_mul_scalar(kb, kvector) + complex_mul_scalar(b, potential);

				SCALAR phase = t * (SCALAR)${detuning} + phi;
				SCALAR sin_phase = (SCALAR)${rabi_freq / 2} * sin(phase);
				SCALAR cos_phase = (SCALAR)${rabi_freq / 2} * cos(phase);

				<%
					# FIXME: remove component hardcoding
					g11 = c.g_by_hbar[(COMP_1_minus1, COMP_1_minus1)]
					g12 = c.g_by_hbar[(COMP_1_minus1, COMP_2_1)]
					g22 = c.g_by_hbar[(COMP_2_1, COMP_2_1)]
				%>

				// FIXME: Some magic here. l111 ~ 10^-42, while single precision float
				// can only handle 10^-38.
				SCALAR temp = n_a * (SCALAR)${1.0e-10};

				*a_res = complex_ctr(ta.y, -ta.x) +
					complex_mul(complex_ctr(
						- temp * temp * (SCALAR)${c.l111 * 1.0e20} - n_b * (SCALAR)${c.l12 / 2},
						- n_a * (SCALAR)${g11} - n_b * (SCALAR)${g12}), a) -
					complex_mul(complex_ctr(sin_phase, cos_phase), b);

				*b_res = complex_ctr(tb.y, -tb.x) +
					complex_mul(complex_ctr(
						- n_a * (SCALAR)${c.l12 / 2} - n_b * (SCALAR)${c.l22 / 2},
						- n_a * (SCALAR)${g12} - n_b * (SCALAR)${g22}), b) -
					complex_mul(complex_ctr(-sin_phase, cos_phase), a);
			}

			EXPORTED_FUNC void calculateRK(GLOBAL_MEM COMPLEX *a, GLOBAL_MEM COMPLEX *b,
				GLOBAL_MEM COMPLEX *a_copy, GLOBAL_MEM COMPLEX *b_copy,
				GLOBAL_MEM COMPLEX *a_kdata, GLOBAL_MEM COMPLEX *b_kdata,
				GLOBAL_MEM COMPLEX *a_res, GLOBAL_MEM COMPLEX *b_res,
				SCALAR t, SCALAR dt,
				GLOBAL_MEM SCALAR *potentials, GLOBAL_MEM SCALAR *kvectors,
				SCALAR phi, int stage)
			{
				DEFINE_INDEXES;

				SCALAR kvector = kvectors[cell_index];
				SCALAR potential = potentials[cell_index];

				COMPLEX ra = a_res[index];
				COMPLEX rb = b_res[index];
				COMPLEX ka = a_kdata[index];
				COMPLEX kb = b_kdata[index];
				COMPLEX a0 = a_copy[index];
				COMPLEX b0 = b_copy[index];

				SCALAR val_coeffs[4] = {0.5, 0.5, 1.0};
				SCALAR res_coeffs[4] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

				COMPLEX a_val, b_val;
				if(stage == 0)
				{
					a_val = a0;
					b_val = b0;
				}
				else
				{
					a_val = ra;
					b_val = rb;
				}

				propagationFunc(&ra, &rb, a_val, b_val, ka, kb, t, dt, kvector, potential, phi);

				if(stage != 3)
				{
					a_res[index] = a0 + complex_mul_scalar(ra, dt * val_coeffs[stage]);
					b_res[index] = b0 + complex_mul_scalar(rb, dt * val_coeffs[stage]);
				}

				a[index] = a[index] + complex_mul_scalar(ra, dt * res_coeffs[stage]);
				b[index] = b[index] + complex_mul_scalar(rb, dt * res_coeffs[stage]);
			}
		"""

		self._program = self._env.compileProgram(kernels, self._constants,
			detuning=self._detuning, rabi_freq=self._rabi_freq,
			COMP_1_minus1=COMP_1_minus1, COMP_2_1=COMP_2_1)
		self._calculateRK = self._program.calculateRK

	def _cpu__propagationFunc(self, a_data, b_data, a_kdata, b_kdata, a_res, b_res, t, dt, phi):

		batch = a_data.size / self._constants.cells
		nvz = self._constants.nvz

		# FIXME: remove hardcoding (g must depend on cloud.a.comp and cloud.b.comp)
		g_by_hbar = self._constants.g_by_hbar
		g11_by_hbar = g_by_hbar[(COMP_1_minus1, COMP_1_minus1)]
		g12_by_hbar = g_by_hbar[(COMP_1_minus1, COMP_2_1)]
		g22_by_hbar = g_by_hbar[(COMP_2_1, COMP_2_1)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		n_a = numpy.abs(a_data) ** 2
		n_b = numpy.abs(b_data) ** 2

		for e in xrange(batch):
			start = e * nvz
			stop = (e + 1) * nvz
			a_res[start:stop,:,:] = -1j * (a_kdata[start:stop,:,:] * self._kvectors +
				a_data[start:stop,:,:] * self._potentials)
			b_res[start:stop,:,:] = -1j * (b_kdata[start:stop,:,:] * self._kvectors +
				b_data[start:stop,:,:] * self._potentials)

		a_res += (n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)) * a_data - \
			0.5j * self._rabi_freq * \
				numpy.exp(1j * (- t * self._detuning - phi)) * b_data

		b_res += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)) * b_data - \
			0.5j * self._rabi_freq * \
				numpy.exp(1j * (t * self._detuning + phi)) * a_data

	def _cpu__calculateRK(self, _, a_data, b_data, a_copy, b_copy, a_kdata, b_kdata,
			a_res, b_res, t, dt, p, k, phi, stage):

		val_coeffs = (0.5, 0.5, 1.0, 0.0)
		res_coeffs = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
		t_coeffs = (0.0, 0.5, 0.5, 1.0)

		if stage == 0:
			a = a_data.copy()
			b = b_data.copy()
		else:
			a = a_res.copy()
			b = b_res.copy()

		self._propagationFunc(a, b, a_kdata, b_kdata, a_res, b_res, t + t_coeffs[stage] * dt, dt, phi)

		a_data += a_res * (dt * res_coeffs[stage])
		b_data += b_res * (dt * res_coeffs[stage])

		a_res[:,:,:] = a_copy + a_res * (dt * val_coeffs[stage])
		b_res[:,:,:] = b_copy + b_res * (dt * val_coeffs[stage])

	def propagate(self, cloud, t, remaining_time):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them

		batch = cloud.a.size / self._constants.cells

		func = self._calculateRK
		fft = self._plan.execute
		cast = self._constants.scalar.cast
		p = self._potentials
		k = self._kvectors

		dt = cast(self._dt)
		t = cast(t)
		phi = cast(self._phi)
		size = cloud.a.size

		self._env.copyBuffer(cloud.a.data, self._a_copy)
		self._env.copyBuffer(cloud.b.data, self._b_copy)

		for i in xrange(4):
			fft(self._a_copy, self._a_kdata, inverse=True, batch=batch)
			fft(self._b_copy, self._b_kdata, inverse=True, batch=batch)
			func(size, cloud.a.data, cloud.b.data,
				self._a_copy, self._b_copy, self._a_kdata, self._b_kdata,
				self._a_res, self._b_res, t, dt, p, k, phi, numpy.int32(i))

		return self._dt

	def run(self, *args, **kwds):
		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		shape = args[0].a.shape
		dtype = args[0].a.dtype

		self._a_copy = self._env.allocate(shape, dtype=dtype)
		self._b_copy = self._env.allocate(shape, dtype=dtype)
		self._a_kdata = self._env.allocate(shape, dtype=dtype)
		self._b_kdata = self._env.allocate(shape, dtype=dtype)
		self._a_res = self._env.allocate(shape, dtype=dtype)
		self._b_res = self._env.allocate(shape, dtype=dtype)

		Evolution.run(self, *args, **kwds)

















class RK5Evolution(Evolution):

	def __init__(self, env, constants, dt=1e-6, eps=1e-9, tiny=1e-3, detuning=0, rabi_freq=0):
		Evolution.__init__(self, env)
		self._constants = constants

		# FIXME: implement adaptive time step propagation
		self._dt = dt
		self._eps = eps
		self._tiny = tiny

		self._detuning = 2 * math.pi * detuning
		self._rabi_freq = 2 * math.pi * rabi_freq

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _propagationFunc(self, state1, state2, t):
		res1 = numpy.empty_like(state1)
		res2 = numpy.empty_like(state2)
		self._propagationFuncInplace(state1, state2, res1, res2, t)
		return res1, res2

	def _propagationFuncInplace(self, state1, state2, res1, res2, t):

		batch = 1 # FIXME: hardcoding
		nvz = self._constants.nvz

		# FIXME: remove hardcoding (g must depend on cloud.a.comp and cloud.b.comp)
		g_by_hbar = self._constants.g_by_hbar
		g11_by_hbar = g_by_hbar[(COMP_1_minus1, COMP_1_minus1)]
		g12_by_hbar = g_by_hbar[(COMP_1_minus1, COMP_2_1)]
		g22_by_hbar = g_by_hbar[(COMP_2_1, COMP_2_1)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		n_a = numpy.abs(state1) ** 2
		n_b = numpy.abs(state2) ** 2

		self._plan.execute(state1, self._a_kdata, batch=batch, inverse=True)
		self._plan.execute(state1, self._b_kdata, batch=batch, inverse=True)

		for e in xrange(batch):
			start = e * nvz
			stop = (e + 1) * nvz
			res1[start:stop,:,:] = -1j * (self._a_kdata[start:stop,:,:] * self._kvectors +
				state1[start:stop,:,:] * self._potentials)
			res2[start:stop,:,:] = -1j * (self._b_kdata[start:stop,:,:] * self._kvectors +
				state2[start:stop,:,:] * self._potentials)

		res1 += (n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)) * state1 - \
			0.5j * self._rabi_freq * \
				numpy.exp(1j * (- t * self._detuning - self._phi)) * state2

		res2 += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)) * state2 - \
			0.5j * self._rabi_freq * \
				numpy.exp(1j * (t * self._detuning + self._phi)) * state1

	def _cpu__propagate_rk5(self, state1, state2, dt, t):

		a = numpy.array([0, 0.2, 0.3, 0.6, 1, 0.875])
		b = numpy.array([
			[0, 0, 0, 0, 0],
			[1.0 / 5, 0, 0, 0, 0],
			[3.0 / 40, 9.0 / 40, 0, 0, 0],
			[3.0 / 10, -9.0 / 10, 6.0 / 5, 0, 0],
			[-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27, 0],
			[1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]
		])
		c = numpy.array([37.0 / 378, 0, 250.0 / 621, 125.0 / 594, 0, 512.0 / 1771])
		cs = numpy.array([2825.0 / 27648, 0, 18575.0 / 48384.0, 13525.0 / 55296, 277.0 / 14336, 0.25])

		k1s1, k1s2 = self._propagationFunc(state1, state2, t)
		k1s1 *= dt
		k1s2 *= dt

		k2s1, k2s2 = self._propagationFunc(
			state1 + b[1,0] * k1s1,
			state2 + b[1,0] * k1s2, t + a[1] * dt)
		k2s1 *= dt
		k2s2 *= dt

		k3s1, k3s2 = self._propagationFunc(
			state1 + b[2,0] * k1s1 + b[2,1] * k2s1,
			state2 + b[2,0] * k1s2 + b[2,1] * k2s2, t + a[2] * dt)
		k3s1 *= dt
		k3s2 *= dt

		k4s1, k4s2 = self._propagationFunc(
			state1 + b[3,0] * k1s1 + b[3,1] * k2s1 + b[3,2] * k3s1,
			state2 + b[3,0] * k1s2 + b[3,1] * k2s2 + b[3,2] * k3s2, t + a[3] * dt)
		k4s1 *= dt
		k4s2 *= dt

		k5s1, k5s2 = self._propagationFunc(
			state1 + b[4,0] * k1s1 + b[4,1] * k2s1 + b[4,2] * k3s1 + b[4,3] * k4s1,
			state2 + b[4,0] * k1s2 + b[4,1] * k2s2 + b[4,2] * k3s2 + b[4,3] * k4s2, t + a[4] * dt)
		k5s1 *= dt
		k5s2 *= dt

		k6s1, k6s2 = self._propagationFunc(
			state1 + b[5,0] * k1s1 + b[5,1] * k2s1 + b[5,2] * k3s1 + b[5,3] * k4s1 + b[5,4] * k5s1,
			state2 + b[5,0] * k1s2 + b[5,1] * k2s2 + b[5,2] * k3s2 + b[5,3] * k4s2 + b[5,4] * k5s2, t + a[5] * dt)
		k6s1 *= dt
		k6s2 *= dt

		y_s1 = state1 + c[0] * k1s1 + c[1] * k2s1 + c[2] * k3s1 + c[3] * k4s1 + c[4] * k5s1 + c[5] * k6s1
		y_s2 = state2 + c[0] * k1s2 + c[1] * k2s2 + c[2] * k3s2 + c[3] * k4s2 + c[4] * k5s2 + c[5] * k6s2

		ys_s1 = state1 + cs[0] * k1s1 + cs[1] * k2s1 + cs[2] * k3s1 + cs[3] * k4s1 + cs[4] * k5s1 + cs[5] * k6s1
		ys_s2 = state2 + cs[0] * k1s2 + cs[1] * k2s2 + cs[2] * k3s2 + cs[3] * k4s2 + cs[4] * k5s2 + cs[5] * k6s2

		delta_s1 = y_s1 - ys_s1
		delta_s2 = y_s2 - ys_s2

		return y_s1, y_s2, numpy.concatenate([delta_s1, delta_s2])

	def _cpu__propagate_rk5_dynamic(self, state1, state2, t):

		safety = 0.9
		eps = self._eps # 1e-9
		tiny = self._tiny # 1e-3

		dt = self._dt

		ds1, ds2 = self._propagationFunc(state1.data, state2.data, t)
		yscal = numpy.concatenate([
			numpy.abs(state1.data) + dt * numpy.abs(ds1),
			numpy.abs(state2.data) + dt * numpy.abs(ds2)
		]) + tiny

		while True:
			#print "Trying with step " + str(dt)
			s1, s2, delta_1 = self._propagate_rk5(state1.data, state2.data, dt, t)
			errmax = numpy.abs(delta_1 / yscal).max() / eps
			#print "Error: " + str(errmax)
			if errmax < 1.0:
				#print "Seems ok"
				break

			# reducing step size and retying step
			dt_temp = safety * dt * (errmax ** (-0.25))
			dt = max(dt_temp, 0.1 * dt)

		dt_used = dt

		if errmax > (5.0 / safety) ** (-1.0 / 0.2):
			self._dt = safety * dt * (errmax ** (-0.2))
		else:
			self._dt = 5.0 * dt

		state1.data.flat[:] = s1.flat
		state2.data.flat[:] = s2.flat

		return dt_used

	def propagate(self, cloud, t, remaining_time):
		return self._propagate_rk5_dynamic(cloud.a, cloud.b, t)

	def run(self, *args, **kwds):
		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		shape = args[0].a.shape
		dtype = args[0].a.dtype

		self._a_kdata = self._env.allocate(shape, dtype=dtype)
		self._b_kdata = self._env.allocate(shape, dtype=dtype)

		Evolution.run(self, *args, **kwds)


















class RK5IPEvolution(Evolution):

	def __init__(self, env, constants, dt=1e-6, eps=1e-6, tiny=1e-3, detuning=0, rabi_freq=0):
		Evolution.__init__(self, env)
		self._constants = constants

		# FIXME: implement adaptive time step propagation
		self._dt = dt
		self._eps = eps
		self._tiny = tiny

		self._detuning = 2 * math.pi * detuning
		self._rabi_freq = 2 * math.pi * rabi_freq

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._dt_times = []
		self._dts = []

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _propagationFunc(self, state1, state2, t, dt):
		res1 = numpy.empty_like(state1)
		res2 = numpy.empty_like(state2)
		self._propagationFuncInplace(state1, state2, res1, res2, t, dt)
		return res1, res2

	def _propagationFuncInplace(self, state1, state2, res1, res2, t, dt):

		batch = 1 # FIXME: hardcoding
		nvz = self._constants.nvz

		# FIXME: remove hardcoding (g must depend on cloud.a.comp and cloud.b.comp)
		g_by_hbar = self._constants.g_by_hbar
		g11_by_hbar = g_by_hbar[(COMP_1_minus1, COMP_1_minus1)]
		g12_by_hbar = g_by_hbar[(COMP_1_minus1, COMP_2_1)]
		g22_by_hbar = g_by_hbar[(COMP_2_1, COMP_2_1)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		x1 = self._a_kdata
		x2 = self._b_kdata

		self._fromIP(state1, state2, x1, x2, dt)

		n_a = numpy.abs(x1) ** 2
		n_b = numpy.abs(x2) ** 2

		for e in xrange(batch):
			start = e * nvz
			stop = (e + 1) * nvz
			res1[start:stop,:,:] = -1j * (x1[start:stop,:,:] * self._potentials)
			res2[start:stop,:,:] = -1j * (x2[start:stop,:,:] * self._potentials)

		res1 += (n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)) * x1 - \
			0.5j * self._rabi_freq * \
				numpy.exp(1j * (- t * self._detuning - self._phi)) * x2

		res2 += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)) * x2 - \
			0.5j * self._rabi_freq * \
				numpy.exp(1j * (t * self._detuning + self._phi)) * x1

		self._toIP(res1, res2, res1, res2, dt)

	def _cpu__propagate_rk5(self, state1, state2, dt, t):

		a = numpy.array([0, 0.2, 0.3, 0.6, 1, 0.875])
		b = numpy.array([
			[0, 0, 0, 0, 0],
			[1.0 / 5, 0, 0, 0, 0],
			[3.0 / 40, 9.0 / 40, 0, 0, 0],
			[3.0 / 10, -9.0 / 10, 6.0 / 5, 0, 0],
			[-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27, 0],
			[1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]
		])
		c = numpy.array([37.0 / 378, 0, 250.0 / 621, 125.0 / 594, 0, 512.0 / 1771])
		cs = numpy.array([2825.0 / 27648, 0, 18575.0 / 48384.0, 13525.0 / 55296, 277.0 / 14336, 0.25])

		k1s1, k1s2 = self._propagationFunc(state1, state2, t, 0)
		k1s1 *= dt
		k1s2 *= dt

		k2s1, k2s2 = self._propagationFunc(
			state1 + b[1,0] * k1s1,
			state2 + b[1,0] * k1s2, t + a[1] * dt, a[1] * dt)
		k2s1 *= dt
		k2s2 *= dt

		k3s1, k3s2 = self._propagationFunc(
			state1 + b[2,0] * k1s1 + b[2,1] * k2s1,
			state2 + b[2,0] * k1s2 + b[2,1] * k2s2, t + a[2] * dt, a[2] * dt)
		k3s1 *= dt
		k3s2 *= dt

		k4s1, k4s2 = self._propagationFunc(
			state1 + b[3,0] * k1s1 + b[3,1] * k2s1 + b[3,2] * k3s1,
			state2 + b[3,0] * k1s2 + b[3,1] * k2s2 + b[3,2] * k3s2, t + a[3] * dt, a[3] * dt)
		k4s1 *= dt
		k4s2 *= dt

		k5s1, k5s2 = self._propagationFunc(
			state1 + b[4,0] * k1s1 + b[4,1] * k2s1 + b[4,2] * k3s1 + b[4,3] * k4s1,
			state2 + b[4,0] * k1s2 + b[4,1] * k2s2 + b[4,2] * k3s2 + b[4,3] * k4s2, t + a[4] * dt, a[4] * dt)
		k5s1 *= dt
		k5s2 *= dt

		k6s1, k6s2 = self._propagationFunc(
			state1 + b[5,0] * k1s1 + b[5,1] * k2s1 + b[5,2] * k3s1 + b[5,3] * k4s1 + b[5,4] * k5s1,
			state2 + b[5,0] * k1s2 + b[5,1] * k2s2 + b[5,2] * k3s2 + b[5,3] * k4s2 + b[5,4] * k5s2, t + a[5] * dt, a[5] * dt)
		k6s1 *= dt
		k6s2 *= dt

		y_s1 = state1 + c[0] * k1s1 + c[1] * k2s1 + c[2] * k3s1 + c[3] * k4s1 + c[4] * k5s1 + c[5] * k6s1
		y_s2 = state2 + c[0] * k1s2 + c[1] * k2s2 + c[2] * k3s2 + c[3] * k4s2 + c[4] * k5s2 + c[5] * k6s2

		ys_s1 = state1 + cs[0] * k1s1 + cs[1] * k2s1 + cs[2] * k3s1 + cs[3] * k4s1 + cs[4] * k5s1 + cs[5] * k6s1
		ys_s2 = state2 + cs[0] * k1s2 + cs[1] * k2s2 + cs[2] * k3s2 + cs[3] * k4s2 + cs[4] * k5s2 + cs[5] * k6s2

		delta_s1 = y_s1 - ys_s1
		delta_s2 = y_s2 - ys_s2

		return y_s1, y_s2, numpy.concatenate([delta_s1, delta_s2])

	def _cpu__propagate_rk5_dynamic(self, state1, state2, t, remaining_time):

		safety = 0.9
		eps = self._eps # 1e-9
		tiny = self._tiny # 1e-3

		dt = self._dt

		ds1, ds2 = self._propagationFunc(state1.data, state2.data, t, 0)
		yscal = numpy.concatenate([
			numpy.abs(state1.data) + dt * numpy.abs(ds1),
			numpy.abs(state2.data) + dt * numpy.abs(ds2)
		]) + tiny

		while True:
			#print "Trying with step " + str(dt)
			s1, s2, delta_1 = self._propagate_rk5(state1.data, state2.data, dt, t)
			errmax = numpy.abs(delta_1 / yscal).max() / eps
			#print "Error: " + str(errmax)
			if errmax < 1.0:
				if dt > remaining_time:
					# Step is fine in terms of error, but bigger then necessary
					dt = remaining_time
					continue
				else:
					#print "Seems ok"
					break

			# reducing step size and retying step
			dt_temp = safety * dt * (errmax ** (-0.25))
			dt = max(dt_temp, 0.1 * dt)

		self._dt_used = dt

		if errmax > (5.0 / safety) ** (-1.0 / 0.2):
			self._dt = safety * dt * (errmax ** (-0.2))
		else:
			self._dt = 5.0 * dt

		state1.data.flat[:] = s1.flat
		state2.data.flat[:] = s2.flat

		self._fromIP(state1.data, state2.data, state1.data, state2.data, self._dt_used)

		#print numpy.sum(numpy.abs(state1.data) ** 2) * self._constants.dV, \
		#	numpy.sum(numpy.abs(state2.data) ** 2) * self._constants.dV

		#raw_input()

		return self._dt_used

	def _toIP(self, s1, s2, res1, res2, dt):
		if dt == 0.0:
			res1.flat[:] = s1.flat[:]
			res2.flat[:] = s2.flat[:]
			return

		self._plan.execute(s1, res1, inverse=True, batch=self._batch)
		self._plan.execute(s2, res2, inverse=True, batch=self._batch)

		kcoeff = numpy.exp(self._kvectors * (1j * dt))
		nvz = self._constants.nvz

		for e in xrange(self._batch):
			start = e * nvz
			stop = (e + 1) * nvz
			res1[start:stop,:,:] *= kcoeff
			res2[start:stop,:,:] *= kcoeff

		self._plan.execute(res1, batch=self._batch)
		self._plan.execute(res2, batch=self._batch)

	def _fromIP(self, s1, s2, res1, res2, dt):
		if dt == 0.0:
			res1.flat[:] = s1.flat[:]
			res2.flat[:] = s2.flat[:]
			return

		self._plan.execute(s1, res1, inverse=True, batch=self._batch)
		self._plan.execute(s2, res2, inverse=True, batch=self._batch)

		kcoeff = numpy.exp(self._kvectors * (-1j * dt))
		nvz = self._constants.nvz

		for e in xrange(self._batch):
			start = e * nvz
			stop = (e + 1) * nvz
			res1[start:stop,:,:] *= kcoeff
			res2[start:stop,:,:] *= kcoeff

		self._plan.execute(res1, batch=self._batch)
		self._plan.execute(res2, batch=self._batch)

	def propagate(self, cloud, t, remaining_time):
		return self._propagate_rk5_dynamic(cloud.a, cloud.b, t, remaining_time)

	def _collectMetrics(self, t):
		self._dts.append(self._dt_used)
		self._dt_times.append(t)

	def getTimeSteps(self):
		return numpy.array(self._dt_times), numpy.array(self._dts)

	def run(self, *args, **kwds):
		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		self._dt_used = 0
		shape = args[0].a.shape
		dtype = args[0].a.dtype

		self._batch = args[0].a.size / self._constants.cells

		self._a_kdata = self._env.allocate(shape, dtype=dtype)
		self._b_kdata = self._env.allocate(shape, dtype=dtype)

		Evolution.run(self, *args, **kwds)
