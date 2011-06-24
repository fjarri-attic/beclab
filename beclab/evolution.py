"""
Classes, modeling the evolution of BEC.
"""

import math
import numpy

from .helpers import *
from .globals import *
from .constants import PSI_FUNC, WIGNER, COMP_1_minus1, COMP_2_1


class TerminateEvolution(Exception):
	pass


class Evolution(PairedCalculation):

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self._env = env

	def _toCanonicalSpace(self, cloud):
		pass

	def _toEvolutionSpace(self, cloud):
		pass

	def _collectMetrics(self, t):
		pass

	def _runCallbacks(self, cloud, callbacks):
		if callbacks is None:
			return

		self._toCanonicalSpace(cloud)
		for callback in callbacks:
			callback(cloud.time, cloud)
		self._toEvolutionSpace(cloud)

	def run(self, cloud, time, callbacks=None, callback_dt=0):

		starting_time = cloud.time
		ending_time = cloud.time + time
		callback_t = 0

		self._toEvolutionSpace(cloud)

		try:
			self._runCallbacks(cloud, callbacks)

			while cloud.time - starting_time < time:
				dt_used = self.propagate(cloud, cloud.time - starting_time, callback_dt - callback_t)
				self._collectMetrics(cloud.time)

				cloud.time += dt_used
				callback_t += dt_used

				if callback_t >= callback_dt:
					self._runCallbacks(cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(cloud, callbacks)

			self._toCanonicalSpace(cloud)

		except TerminateEvolution:
			return cloud.time


class SplitStepEvolution(Evolution):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants, rabi_freq=0, detuning=0, dt=None, noise=True):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		# FIXME: temporary stub; remove when implement constants/grid separation
		self._dt = dt if dt is not None else self._constants.dt_evo
		self._noise = noise

		self._detuning = 2 * math.pi * detuning
		self._rabi_freq = 2 * math.pi * rabi_freq

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
		self._random = createRandom(env, constants.double)

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._projector_mask = getProjectorMask(self._env, self._constants)

		self._prepare()

	def _cpu__projector(self, cloud):
		nvz = self._constants.nvz
		a_data = cloud.a.data
		b_data = cloud.b.data

		for e in xrange(self._constants.ensembles):
			start = e * nvz
			stop = (e + 1) * nvz
			if self._constants.dim == 3:
				a_data[start:stop,:,:] *= self._projector_mask
				b_data[start:stop,:,:] *= self._projector_mask
			else:
				a_data[start:stop] *= self._projector_mask
				b_data[start:stop] *= self._projector_mask

	def _gpu__projector(self, cloud):
		self._projector_func(cloud.a.size, cloud.a.data, cloud.b.data, self._projector_mask)

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):

		kernels = """
			<%!
				from math import sqrt, pi
			%>

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			EXPORTED_FUNC void propagateKSpaceRealTime(GLOBAL_MEM COMPLEX *a, GLOBAL_MEM COMPLEX *b,
				SCALAR dt, GLOBAL_MEM SCALAR *kvectors)
			{
				DEFINE_INDEXES;

				SCALAR kvector = kvectors[cell_index];
				SCALAR prop_angle = -kvector * dt / 2;
				COMPLEX prop_coeff = complex_ctr(cos(prop_angle), sin(prop_angle));

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				a[index] = complex_mul(a0, prop_coeff);
				b[index] = complex_mul(b0, prop_coeff);
			}

			// Propagates state vector in x-space for evolution calculation
			EXPORTED_FUNC void propagateXSpaceTwoComponent(GLOBAL_MEM COMPLEX *aa,
				GLOBAL_MEM COMPLEX *bb, SCALAR dt, SCALAR t, GLOBAL_MEM SCALAR *potentials,
				SCALAR phi)
			{
				DEFINE_INDEXES;

				SCALAR V = potentials[cell_index];

				COMPLEX a = aa[index];
				COMPLEX b = bb[index];

				//store initial x-space field
				COMPLEX a0 = a;
				COMPLEX b0 = b;

				COMPLEX N1, N2;
				SCALAR n_a, n_b;

				%if rabi_freq == 0.0:
					COMPLEX da, db;
				%else:
					SCALAR k, f;
					COMPLEX rt, l1, l2, ev10, ev11, ev_inv_coeff, l1_exp, l2_exp;
					COMPLEX m[4];
				%endif

				<%
					# FIXME: remove component hardcoding
					g11 = c.g_by_hbar[(COMP_1_minus1, COMP_1_minus1)]
					g12 = c.g_by_hbar[(COMP_1_minus1, COMP_2_1)]
					g22 = c.g_by_hbar[(COMP_2_1, COMP_2_1)]
				%>

				SCALAR temp;

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					n_a = squared_abs(a);
					n_b = squared_abs(b);

					// FIXME: Some magic here. l111 ~ 10^-42, while single precision float
					// can only handle 10^-38.
					temp = n_a * (SCALAR)${1.0e-10};

					N1 = complex_ctr(
						-(temp * temp * (SCALAR)${c.l111 * 1e20} + (SCALAR)${c.l12} * n_b) / 2,
						-(V + (SCALAR)${g11} * n_a + (SCALAR)${g12} * n_b));
					N2 = complex_ctr(
						-((SCALAR)${c.l22} * n_b + (SCALAR)${c.l12} * n_a) / 2,
						-(V + (SCALAR)${g22} * n_b + (SCALAR)${g12} * n_a));

					%if rabi_freq == 0.0:

						// calculate midpoint log derivative and exponentiate
						da = cexp(complex_mul_scalar(N1, (dt / 2)));
						db = cexp(complex_mul_scalar(N2, (dt / 2)));

						//propagate to midpoint using log derivative
						a = complex_mul(a0, da);
						b = complex_mul(b0, db);

					%else:

						k = (SCALAR)${rabi_freq};
						f = (SCALAR)${detuning} * t + phi;

						// calculating exp([[N1, -ik exp(-if)/2], [-ik exp(if)/2, N2]])
						rt = csqrt(
							complex_ctr(-k * k, 0) +
							complex_mul(N1 - N2, N1 - N2));

						l1 = complex_mul_scalar(N1 + N2 - rt, 0.5); // eigenvalues 1
						l2 = complex_mul_scalar(rt + N1 + N2, 0.5); // eigenvalues 2

						// elements of eigenvector matrix ([1, 1], [ev10, ev11])
						ev10 = complex_mul(
							cexp((SCALAR)-1.0 / k, f + (SCALAR)${pi / 2}),
							rt + N1 - N2);
						ev11 = complex_mul(
							cexp((SCALAR)1.0 / k, f + (SCALAR)${pi / 2}),
							rt - N1 + N2);

						// elements of inverse eigenvector matrix
						// ([-ev11, 1], [ev10, -1]) * ev_inf_coeff
						ev_inv_coeff = complex_div(complex_ctr(1, 0), ev10 - ev11);

						l1_exp = cexp(complex_mul_scalar(l1, dt / 2));
						l2_exp = cexp(complex_mul_scalar(l2, dt / 2));

						m[0] = complex_mul(
							complex_mul(l2_exp, ev10) - complex_mul(l1_exp, ev11),
							ev_inv_coeff);
						m[1] = complex_mul(l1_exp - l2_exp, ev_inv_coeff);
						m[2] = complex_mul(
							complex_mul(complex_mul(ev10, ev11), l2_exp - l1_exp),
							ev_inv_coeff);
						m[3] = complex_mul(
							complex_mul(l1_exp, ev10) - complex_mul(l2_exp, ev11),
							ev_inv_coeff);

						//propagate to midpoint
						a = complex_mul(a0, m[0]) + complex_mul(b0, m[1]);
						b = complex_mul(a0, m[2]) + complex_mul(b0, m[3]);

					%endif

				%endfor

				//propagate to endpoint using log derivative
				%if rabi_freq == 0.0:
					aa[index] = complex_mul(a, da);
					bb[index] = complex_mul(b, db);
				%else:
					aa[index] = complex_mul(a, m[0]) + complex_mul(b, m[1]);
					bb[index] = complex_mul(a, m[2]) + complex_mul(b, m[3]);
				%endif
			}

			INTERNAL_FUNC void noiseFunc(COMPLEX *G, COMPLEX a0, COMPLEX b0, SCALAR dt)
			{
				<%
					coeff = sqrt(1.0 / c.dV)
					k111 = c.l111 / 6.0
					k12 = c.l12 / 2.0
					k22 = c.l22 / 4.0
				%>

				COMPLEX gamma111_1 = complex_mul_scalar(complex_mul(a0, a0),
					(SCALAR)${sqrt(k111) * 3.0 * coeff});
				COMPLEX gamma12_1 = complex_mul_scalar(b0,
					(SCALAR)${sqrt(k12) * coeff});
				COMPLEX gamma12_2 = complex_mul_scalar(a0,
					(SCALAR)${sqrt(k12) * coeff});
				COMPLEX gamma22_2 = complex_mul_scalar(b0,
					(SCALAR)${sqrt(k22) * 2.0 * coeff});

				G[0] = gamma111_1;
				G[1] = gamma12_1;
				G[2] = complex_ctr(0, 0);
				G[3] = complex_ctr(0, 0);
				G[4] = gamma12_2;
				G[5] = gamma22_2;
			}

			EXPORTED_FUNC void addNoise(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b, SCALAR dt,
				GLOBAL_MEM COMPLEX *randoms)
			{
				DEFINE_INDEXES;
				COMPLEX G[6];
				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				%for i in xrange(6):
				COMPLEX Z${i} = randoms[index + ${c.cells * c.ensembles * i}];
				%endfor

				SCALAR sdt2 = sqrt(dt / (SCALAR)2.0);
				SCALAR sdt = sqrt(dt);

				noiseFunc(G, a0, b0, dt);

				noiseFunc(G,
					a0 + complex_mul_scalar(
						complex_mul(G[0], Z0) +
						complex_mul(G[1], Z1) +
						complex_mul(G[2], Z2), sdt2),
					b0 + complex_mul_scalar(
						complex_mul(G[3], Z0) +
						complex_mul(G[4], Z1) +
						complex_mul(G[5], Z2), sdt2), dt);

				a[index] = a0 + complex_mul_scalar(
						complex_mul(G[0], Z3) +
						complex_mul(G[1], Z4) +
						complex_mul(G[2], Z5), sdt);

				b[index] = b0 + complex_mul_scalar(
						complex_mul(G[3], Z3) +
						complex_mul(G[4], Z4) +
						complex_mul(G[5], Z5), sdt);
			}

			EXPORTED_FUNC void projector(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b, GLOBAL_MEM SCALAR *projector_mask)
			{
				DEFINE_INDEXES;
				SCALAR mask_val = projector_mask[cell_index];

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				a[index] = complex_mul_scalar(a0, mask_val);
				b[index] = complex_mul_scalar(b0, mask_val);
			}
		"""

		self._program = self._env.compileProgram(kernels, self._constants,
			COMP_1_minus1=COMP_1_minus1, COMP_2_1=COMP_2_1,
			rabi_freq=self._rabi_freq, detuning=self._detuning)
		self._kpropagate_func = self._program.propagateKSpaceRealTime
		self._xpropagate_func = self._program.propagateXSpaceTwoComponent
		self._addnoise_func = self._program.addNoise
		self._projector_func = self._program.projector

	def _toKSpace(self, cloud):
		batch = cloud.a.size / self._constants.cells
		self._plan.execute(cloud.a.data, batch=batch, inverse=True)
		self._plan.execute(cloud.b.data, batch=batch, inverse=True)

	def _toXSpace(self, cloud):
		batch = cloud.a.size / self._constants.cells
		self._plan.execute(cloud.a.data, batch=batch)
		self._plan.execute(cloud.b.data, batch=batch)

	def _gpu__kpropagate(self, cloud, dt):
		self._kpropagate_func(cloud.a.size,
			cloud.a.data, cloud.b.data, self._constants.scalar.cast(dt), self._kvectors)

	def _cpu__kpropagate(self, cloud, dt):
		kcoeff = numpy.exp(self._kvectors * (-1j * dt / 2))
		data1 = cloud.a.data
		data2 = cloud.b.data
		nvz = self._constants.nvz

		if self._constants.dim == 1:
			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				data1[start:stop] *= kcoeff
				data2[start:stop] *= kcoeff
		else:
			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				data1[start:stop,:,:] *= kcoeff
				data2[start:stop,:,:] *= kcoeff

	def _gpu__xpropagate(self, cloud, dt, t):
		cast = self._constants.scalar.cast
		self._xpropagate_func(cloud.a.size, cloud.a.data, cloud.b.data,
			cast(dt), cast(t), self._potentials, cast(self._phi))

	def _cpu__xpropagate(self, cloud, dt, t):
		a = cloud.a
		b = cloud.b
		a0 = a.data.copy()
		b0 = b.data.copy()

		comp1 = cloud.a.comp
		comp2 = cloud.b.comp
		g_by_hbar = self._constants.g_by_hbar
		g11_by_hbar = g_by_hbar[(comp1, comp1)]
		g12_by_hbar = g_by_hbar[(comp1, comp2)]
		g22_by_hbar = g_by_hbar[(comp2, comp2)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		p = -self._potentials * 1j
		nvz = self._constants.nvz

		m = numpy.empty((2, 2, a.data.size), dtype=a.data.dtype)

		for iter in xrange(self._constants.itmax):
			n_a = numpy.abs(a.data) ** 2
			n_b = numpy.abs(b.data) ** 2

			N1 = n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) - \
				1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)

			N2 = n_b * (-l22 / 2) + n_a * (-l12 / 2) - \
				1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)

			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				N1[start:stop] += p
				N2[start:stop] += p

			if self._rabi_freq == 0.0:

				da = numpy.exp(N1 * (dt / 2))
				db = numpy.exp(N2 * (dt / 2))

				a.data = a0 * da
				b.data = b0 * db

			else:

				k = self._rabi_freq
				f = self._detuning * t + self._phi

				# calculating exp([[N1, -ik exp(-if)/2], [-ik exp(if)/2, N2]])
				rt = numpy.sqrt(-k ** 2 + (N1 - N2) ** 2 + 0j)

				l1 = 0.5 * (-rt + N1 + N2) # eigenvalues 1
				l2 = 0.5 * (rt + N1 + N2) # eigenvalues 2

				#ev = numpy.array([
				#	[1, -(1j * numpy.exp(1j * f) * (rt + N1 - N2)) / k],
				#	[1, (1j * numpy.exp(1j * f) * (rt - N1 + N2)) / k]
				#])

				# elements of eigenvector matrix ([1, 1], [ev10, ev11])
				ev10 = -(1j * numpy.exp(1j * f) * (rt + N1 - N2)) / k
				ev11 = (1j * numpy.exp(1j * f) * (rt - N1 + N2)) / k

				# elements of inverse eigenvector matrix
				# ([-ev11, 1], [ev10, -1]) / ev_inf_coeff
				ev_inv_coeff = ev10 - ev11

				l1_exp = numpy.exp(l1 * dt / 2)
				l2_exp = numpy.exp(l2 * dt / 2)

				m[0,0,:] = ((l2_exp * ev10 - l1_exp * ev11) / ev_inv_coeff).flatten()
				m[0,1,:] = ((l1_exp - l2_exp) / ev_inv_coeff).flatten()
				m[1,0,:] = ((ev10 * ev11 * (l2_exp - l1_exp)) / ev_inv_coeff).flatten()
				m[1,1,:] = ((l1_exp * ev10 - l2_exp * ev11) / ev_inv_coeff).flatten()

				a.data = m[0,0,:].reshape(a0.shape) * a0 + m[0,1,:].reshape(a0.shape) * b0
				b.data = m[1,0,:].reshape(a0.shape) * a0 + m[1,1,:].reshape(a0.shape) * b0

		if self._rabi_freq == 0.0:
			a.data *= da
			b.data *= db
		else:
			a_copy = a.data.copy()
			a.data = m[0,0,:].reshape(a0.shape) * a.data + m[0,1,:].reshape(a0.shape) * b.data
			b.data = m[1,0,:].reshape(a0.shape) * a_copy + m[1,1,:].reshape(a0.shape) * b.data

	def _noiseFunc(self, a_data, b_data):
		coeff = math.sqrt(1.0 / self._constants.dV)
		k111 = self._constants.l111 / 6.0
		k12 = self._constants.l12 / 2.0
		k22 = self._constants.l22 / 4.0

		gamma111_1 = a_data ** 2 * (math.sqrt(k111) * 3.0 * coeff)
		gamma12_1 = b_data * (math.sqrt(k12) * coeff)
		gamma12_2 = a_data * (math.sqrt(k12) * coeff)
		gamma22_2 = b_data * (math.sqrt(k22) * 2.0 * coeff)

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)

		return gamma111_1, gamma12_1, zeros, zeros, gamma12_2, gamma22_2

	def _cpu__propagateNoise(self, cloud, dt):
		shape = self._constants.ens_shape
		r_shape = tuple([6] + list(shape))

		randoms = self._random.random_normal(size=r_shape)

		G00, G01, G02, G03, G04, G05 = self._noiseFunc(cloud.a.data, cloud.b.data)
		G10, G11, G12, G13, G14, G15 = self._noiseFunc(
			cloud.a.data + math.sqrt(dt / 2.0) * (
				G00 * randoms[0] + G01 * randoms[1] + G02 * randoms[2]),
			cloud.b.data + math.sqrt(dt / 2.0) * (
				G03 * randoms[0] + G04 * randoms[1] + G05 * randoms[2])
		)

		cloud.a.data += math.sqrt(dt) * (G10 * randoms[3] +
			G11 * randoms[4] + G12 * randoms[5])
		cloud.b.data += math.sqrt(dt) * (G13 * randoms[3] +
			G14 * randoms[4] + G15 * randoms[5])

	def _gpu__propagateNoise(self, cloud, dt):
		randoms = self._random.random_normal(size=cloud.a.size * 6)

		self._addnoise_func(cloud.a.size, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(dt), randoms)

	def _finishStep(self, cloud):
		if self._midstep:
			self._kpropagate(cloud, self._dt)
			self._midstep = False

	def _toCanonicalSpace(self, cloud):
		self._finishStep(cloud)
		self._toXSpace(cloud)

	def _toEvolutionSpace(self, cloud):
		self._toKSpace(cloud)

	def propagate(self, cloud, t, remaining_time):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them
		if self._midstep:
			self._kpropagate(cloud, self._dt * 2)
		else:
			self._kpropagate(cloud, self._dt)

		if cloud.type == WIGNER and noise:
			self._projector(cloud)

		self._toXSpace(cloud)
		self._xpropagate(cloud, self._dt, t)

		if cloud.type == WIGNER and noise:
			self._propagateNoise(cloud, self._dt)

		self._midstep = True
		self._toKSpace(cloud)

		return self._dt

	def run(self, *args, **kwds):
		if 'starting_phase' in kwds:
			starting_phase = kwds.pop('starting_phase')
			self._phi = starting_phase
		else:
			self._phi = 0.0

		Evolution.run(self, *args, **kwds)


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
