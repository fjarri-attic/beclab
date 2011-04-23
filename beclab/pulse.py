import math
import copy
from .helpers import *
from .globals import *
from .constants import PSI_FUNC, WIGNER, COMP_1_minus1, COMP_2_1
from .evolution import SplitStepEvolution2


class Pulse(PairedCalculation):

	def __init__(self, env, constants, detuning=None, starting_phase=0, dt=None):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		if detuning is None:
			self._detuning = self._constants.w_detuning
		else:
			self._detuning = 2 * math.pi * detuning

		self._starting_phase = starting_phase

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		c = copy.deepcopy(constants)
		c.dt_evo = c.t_rabi / 1e3 if dt is None else dt
		self._evolution = SplitStepEvolution2(env, c,
			rabi_freq=c.w_rabi / 2.0 / math.pi,
			detuning=self._detuning / 2.0 / math.pi)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernels = """
			<%!
				from math import sqrt
			%>

			EXPORTED_FUNC void calculateMatrix(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b,
				SCALAR theta, SCALAR phi)
			{
				DEFINE_INDEXES;

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				SCALAR sin_half_theta = sin(theta / (SCALAR)2.0);
				SCALAR cos_half_theta = cos(theta / (SCALAR)2.0);

				COMPLEX minus_i = complex_ctr(0, -1);

				COMPLEX k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, -phi))
				), sin_half_theta);

				COMPLEX k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, phi))
				), sin_half_theta);

				a[index] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				b[index] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}

			EXPORTED_FUNC void calculateNoiseMatrix(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b, SCALAR *thetas, SCALAR *phis)
			{
				DEFINE_INDEXES;

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				int trajectory = index / ${c.cells};
				SCALAR theta = thetas[trajectory];
				SCALAR phi = phis[trajectory];

				SCALAR sin_half_theta = sin(theta / (SCALAR)2.0);
				SCALAR cos_half_theta = cos(theta / (SCALAR)2.0);

				COMPLEX minus_i = complex_ctr(0, -1);

				COMPLEX k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, -phi))
				), sin_half_theta);

				COMPLEX k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, phi))
				), sin_half_theta);

				a[index] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				b[index] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}
		"""

		self._program = self._env.compileProgram(kernels, self._constants,
			detuning=self._detuning, COMP_1_minus1=COMP_1_minus1, COMP_2_1=COMP_2_1)
		self._calculateMatrix = self._program.calculateMatrix
		self._calculateNoiseMatrixFunc = self._program.calculateNoiseMatrix

	def _cpu__applyMatrix(self, cloud, theta, phi):
		a = cloud.a.data
		b = cloud.b.data

		a0 = a.copy()
		b0 = b.copy()

		half_theta = theta / 2.0
		k1 = self._constants.scalar.cast(math.cos(half_theta))
		k2 = self._constants.complex.cast(-1j * numpy.exp(-1j * phi) * math.sin(half_theta))
		k3 = self._constants.complex.cast(-1j * numpy.exp(1j * phi) * math.sin(half_theta))

		if self._constants.dim == 1:
			a[:] = a0 * k1 + b0 * k2
			b[:] = a0 * k3 + b0 * k1
		else:
			a[:,:,:] = a0 * k1 + b0 * k2
			b[:,:,:] = a0 * k3 + b0 * k1

	def _applyNoiseMatrix(self, cloud, theta, phi, theta_noise, phi_noise):

		ens = self._constants.ensembles
		dtype = self._constants.scalar.dtype

		if phi_noise > 0.0:
			phis = numpy.random.normal(size=(ens,), scale=phi_noise, loc=phi)
		else:
			phis = numpy.ones(ens) * phi

		if theta_noise > 0.0:
			thetas = numpy.random.normal(size=(ens,), scale=theta_noise, loc=theta)
		else:
			thetas = numpy.ones(ens) * theta

		d_phis = self._env.toDevice(phis.astype(dtype))
		d_thetas = self._env.toDevice(thetas.astype(dtype))

		self._calculateNoiseMatrix(cloud.a.data, cloud.b.data, d_thetas, d_phis)

	def _gpu__calculateNoiseMatrix(self, a_data, b_data, thetas, phis):
		return self._calculateNoiseMatrixFunc(a_data.size, a_data, b_data, thetas, phis)

	def _cpu__calculateNoiseMatrix(self, a, b, thetas, phis):

		ens = self._constants.ensembles

		a0 = a.copy()
		b0 = b.copy()

		half_thetas = thetas / 2.0
		k1 = self._constants.scalar.cast(numpy.cos(half_thetas))
		k2 = self._constants.complex.cast(-1j * numpy.exp(-1j * phis) * numpy.sin(half_thetas))
		k3 = self._constants.complex.cast(-1j * numpy.exp(1j * phis) * numpy.sin(half_thetas))

		a_view = a.ravel()
		b_view = b.ravel()
		a0_view = a0.ravel()
		b0_view = b0.ravel()
		step = self._constants.cells

		for e in xrange(ens):
			start = e * step
			stop = (e + 1) * step

			a0_part = a0_view[start:stop]
			b0_part = b0_view[start:stop]

			a_view[start:stop] = a0_part * k1[e] + b0_part * k2[e]
			b_view[start:stop] = a0_part * k3[e] + b0_part * k1[e]

	def _gpu__applyMatrix(self, cloud, theta, phi):
		self._calculateMatrix(cloud.a.size, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(theta),
			self._constants.scalar.cast(phi))

	def apply(self, cloud, theta, matrix=True, theta_noise=0.0, phi_noise=0.0):
		phi = cloud.time * self._detuning + self._starting_phase

		# FIXME: should we change cloud time depending on pulse time?
		# (if theta_noise != 0)
		t_pulse = (theta / math.pi / 2.0) * self._constants.t_rabi

		if phi_noise > 0 or theta_noise > 0:
			self._applyNoiseMatrix(cloud, theta, phi, theta_noise, phi_noise)
		elif matrix:
			self._applyMatrix(cloud, theta, phi)
		else:
			self._evolution.run(cloud, t_pulse, starting_phase=phi)

		cloud.time += t_pulse


class RK4Pulse(PairedCalculation):

	def __init__(self, env, constants, detuning=None, starting_phase=0):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		if detuning is None:
			self._detuning = self._constants.w_detuning
		else:
			self._detuning = 2 * math.pi * detuning

		self._starting_phase = starting_phase

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernels = """
			<%!
				from math import sqrt
			%>

			EXPORTED_FUNC void calculateMatrix(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b,
				SCALAR theta, SCALAR phi)
			{
				DEFINE_INDEXES;

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				SCALAR sin_half_theta = sin(theta / (SCALAR)2.0);
				SCALAR cos_half_theta = cos(theta / (SCALAR)2.0);

				COMPLEX minus_i = complex_ctr(0, -1);

				COMPLEX k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, -phi))
				), sin_half_theta);

				COMPLEX k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, phi))
				), sin_half_theta);

				a[index] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				b[index] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}

			EXPORTED_FUNC void calculateNoiseMatrix(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b, SCALAR *thetas, SCALAR *phis)
			{
				DEFINE_INDEXES;

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				int trajectory = index / ${c.cells};
				SCALAR theta = thetas[trajectory];
				SCALAR phi = phis[trajectory];

				SCALAR sin_half_theta = sin(theta / (SCALAR)2.0);
				SCALAR cos_half_theta = cos(theta / (SCALAR)2.0);

				COMPLEX minus_i = complex_ctr(0, -1);

				COMPLEX k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, -phi))
				), sin_half_theta);

				COMPLEX k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, phi))
				), sin_half_theta);

				a[index] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				b[index] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}

			INTERNAL_FUNC void propagationFunc(COMPLEX *a_res, COMPLEX *b_res,
				COMPLEX a, COMPLEX b,
				COMPLEX ka, COMPLEX kb,
				SCALAR t, SCALAR dt,
				SCALAR kvector, SCALAR potential,
				SCALAR phi)
			{
				SCALAR n_a = squared_abs(a);
				SCALAR n_b = squared_abs(b);

				COMPLEX ta = complex_mul_scalar(ka, kvector) - complex_mul_scalar(a, potential);
				COMPLEX tb = complex_mul_scalar(kb, kvector) - complex_mul_scalar(b, potential);

				SCALAR phase = t * (SCALAR)${detuning} + phi;
				SCALAR sin_phase = ${c.w_rabi / 2} * sin(phase);
				SCALAR cos_phase = ${c.w_rabi / 2} * cos(phase);

				<%
					# FIXME: remove component hardcoding
					g11 = c.g_by_hbar[(COMP_1_minus1, COMP_1_minus1)]
					g12 = c.g_by_hbar[(COMP_1_minus1, COMP_2_1)]
					g22 = c.g_by_hbar[(COMP_2_1, COMP_2_1)]
				%>

				// FIXME: Some magic here. l111 ~ 10^-42, while single precision float
				// can only handle 10^-38.
				SCALAR temp = n_a * ${1.0e-10};

				*a_res = complex_ctr(-ta.y, ta.x) +
					complex_mul(complex_ctr(
						- temp * temp * ${c.l111 * 1.0e20} - n_b * ${c.l12 / 2},
						- n_a * ${g11} - n_b * ${g12}), a) -
					complex_mul(complex_ctr(sin_phase, cos_phase), b);

				*b_res = complex_ctr(-tb.y, tb.x) +
					complex_mul(complex_ctr(
						- n_a * ${c.l12 / 2} - n_b * ${c.l22 / 2},
						- n_a * ${g12} - n_b * ${g22}), b) -
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

				if(stage != 4)
				{
					a_res[index] = a0 + complex_mul_scalar(ra, dt * val_coeffs[stage]);
					b_res[index] = b0 + complex_mul_scalar(rb, dt * val_coeffs[stage]);
				}

				a[index] = a[index] + complex_mul_scalar(ra, dt * res_coeffs[stage]);
				b[index] = b[index] + complex_mul_scalar(rb, dt * res_coeffs[stage]);
			}
		"""

		self._program = self._env.compileProgram(kernels, self._constants,
			detuning=self._detuning, COMP_1_minus1=COMP_1_minus1, COMP_2_1=COMP_2_1)
		self._calculateMatrix = self._program.calculateMatrix
		self._calculateNoiseMatrixFunc = self._program.calculateNoiseMatrix
		self._calculateRK = self._program.calculateRK

	def _cpu__applyMatrix(self, cloud, theta, phi):
		a = cloud.a.data
		b = cloud.b.data

		a0 = a.copy()
		b0 = b.copy()

		half_theta = theta / 2.0
		k1 = self._constants.scalar.cast(math.cos(half_theta))
		k2 = self._constants.complex.cast(-1j * numpy.exp(-1j * phi) * math.sin(half_theta))
		k3 = self._constants.complex.cast(-1j * numpy.exp(1j * phi) * math.sin(half_theta))

		if self._constants.dim == 1:
			a[:] = a0 * k1 + b0 * k2
			b[:] = a0 * k3 + b0 * k1
		else:
			a[:,:,:] = a0 * k1 + b0 * k2
			b[:,:,:] = a0 * k3 + b0 * k1

	def _applyNoiseMatrix(self, cloud, theta, phi, theta_noise, phi_noise):

		ens = self._constants.ensembles
		dtype = self._constants.scalar.dtype

		if phi_noise > 0.0:
			phis = numpy.random.normal(size=(ens,), scale=phi_noise, loc=phi)
		else:
			phis = numpy.ones(ens) * phi

		if theta_noise > 0.0:
			thetas = numpy.random.normal(size=(ens,), scale=theta_noise, loc=theta)
		else:
			thetas = numpy.ones(ens) * theta

		d_phis = self._env.toDevice(phis.astype(dtype))
		d_thetas = self._env.toDevice(thetas.astype(dtype))

		self._calculateNoiseMatrix(cloud.a.data, cloud.b.data, d_thetas, d_phis)

	def _gpu__calculateNoiseMatrix(self, a_data, b_data, thetas, phis):
		return self._calculateNoiseMatrixFunc(a_data.size, a_data, b_data, thetas, phis)

	def _cpu__calculateNoiseMatrix(self, a, b, thetas, phis):

		ens = self._constants.ensembles

		a0 = a.copy()
		b0 = b.copy()

		half_thetas = thetas / 2.0
		k1 = self._constants.scalar.cast(numpy.cos(half_thetas))
		k2 = self._constants.complex.cast(-1j * numpy.exp(-1j * phis) * numpy.sin(half_thetas))
		k3 = self._constants.complex.cast(-1j * numpy.exp(1j * phis) * numpy.sin(half_thetas))

		a_view = a.ravel()
		b_view = b.ravel()
		a0_view = a0.ravel()
		b0_view = b0.ravel()
		step = self._constants.cells

		for e in xrange(ens):
			start = e * step
			stop = (e + 1) * step

			a0_part = a0_view[start:stop]
			b0_part = b0_view[start:stop]

			a_view[start:stop] = a0_part * k1[e] + b0_part * k2[e]
			b_view[start:stop] = a0_part * k3[e] + b0_part * k1[e]

	def _gpu__applyMatrix(self, cloud, theta, phi):
		self._calculateMatrix(cloud.a.size, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(theta),
			self._constants.scalar.cast(phi))

	def _cpu__calculateRK(self, _, a_data, b_data, a_copy, b_copy, a_kdata, b_kdata,
			a_res, b_res, t, dt, p, k, phi, stage):

		val_coeffs = (0.5, 0.5, 1.0, 0.0)
		res_coeffs = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)

		if stage == 0:
			a = a_data.copy()
			b = b_data.copy()
		else:
			a = a_res.copy()
			b = b_res.copy()

		self._propagationFunc(a, b, a_kdata, b_kdata, a_res, b_res, t, dt, phi)

		a_data += a_res * (dt * res_coeffs[stage])
		b_data += b_res * (dt * res_coeffs[stage])

		a_res[:,:,:] = a_copy + a_res * (dt * val_coeffs[stage])
		b_res[:,:,:] = b_copy + b_res * (dt * val_coeffs[stage])

	def _applyReal(self, cloud, t_pulse, phi):

		batch = cloud.a.size / self._constants.cells
		shape = cloud.a.shape

		func = self._calculateRK
		fft = self._plan.execute
		cast = self._constants.scalar.cast
		p = self._potentials
		k = self._kvectors

		steps = 50
		dt = cast(t_pulse / steps)
		phi = cast(phi)

		size = cloud.a.size
		dtype = cloud.a.dtype

		a_copy = self._env.allocate(shape, dtype=dtype)
		b_copy = self._env.allocate(shape, dtype=dtype)
		a_kdata = self._env.allocate(shape, dtype=dtype)
		b_kdata = self._env.allocate(shape, dtype=dtype)
		a_res = self._env.allocate(shape, dtype=dtype)
		b_res = self._env.allocate(shape, dtype=dtype)

		for i in xrange(steps):
			t = cast(dt * i)

			self._env.copyBuffer(cloud.a.data, a_copy)
			self._env.copyBuffer(cloud.b.data, b_copy)

			fft(a_copy, a_kdata, inverse=True, batch=batch)
			fft(b_copy, b_kdata, inverse=True, batch=batch)
			func(size, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(0))

			fft(a_res, a_kdata, inverse=True, batch=batch)
			fft(b_res, b_kdata, inverse=True, batch=batch)
			func(size, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(1))

			fft(a_res, a_kdata, inverse=True, batch=batch)
			fft(b_res, b_kdata, inverse=True, batch=batch)
			func(size, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(2))

			fft(a_res, a_kdata, inverse=True, batch=batch)
			fft(b_res, b_kdata, inverse=True, batch=batch)
			func(size, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(3))

	def _propagationFunc(self, a_data, b_data, a_kdata, b_kdata, a_res, b_res, t, dt, phi):

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
			a_res[start:stop,:,:] = 1j * (a_kdata[start:stop,:,:] * self._kvectors -
				a_data[start:stop,:,:] * self._potentials)
			b_res[start:stop,:,:] = 1j * (b_kdata[start:stop,:,:] * self._kvectors -
				b_data[start:stop,:,:] * self._potentials)

		a_res += (n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)) * a_data - \
			0.5j * self._constants.w_rabi * \
				numpy.exp(1j * (- t * self._detuning - phi)) * b_data

		b_res += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)) * b_data - \
			0.5j * self._constants.w_rabi * \
				numpy.exp(1j * (t * self._detuning + phi)) * a_data

	def apply(self, cloud, theta, matrix=True, theta_noise=0.0, phi_noise=0.0):
		phi = cloud.time * self._detuning + self._starting_phase

		# FIXME: should we change cloud time depending on pulse time?
		# (if theta_noise != 0)
		t_pulse = (theta / math.pi / 2.0) * self._constants.t_rabi

		if phi_noise > 0 or theta_noise > 0:
			self._applyNoiseMatrix(cloud, theta, phi, theta_noise, phi_noise)
		elif matrix:
			self._applyMatrix(cloud, theta, phi)
		else:
			self._applyReal(cloud, t_pulse, phi)

		cloud.time += t_pulse
