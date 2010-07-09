"""
Classes, modeling the evolution of BEC.
"""

import math
import copy
from mako.template import Template
import numpy

try:
	import pyopencl as cl
except:
	pass

from .globals import *
from .fft import createPlan
from .reduce import getReduce
from .ground_state import GPEGroundState
from .constants import PSI_FUNC, WIGNER


class TerminateEvolution(Exception):
	pass


class Pulse(PairedCalculation):

	def __init__(self, env, constants, detuning=None, starting_phase=0):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		if detuning is None:
			self._detuning = self._constants.detuning
		else:
			self._detuning = 2 * math.pi * detuning / self._constants.w_rho

		self._starting_phase = starting_phase

		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)

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

			__kernel void calculateMatrix(__global ${c.complex.name} *a,
				__global ${c.complex.name} *b,
				${c.scalar.name} theta, ${c.scalar.name} phi)
			{
				DEFINE_INDEXES;

				${c.complex.name} a0 = a[index];
				${c.complex.name} b0 = b[index];

				${c.scalar.name} sin_half_theta = sin(theta / 2);
				${c.scalar.name} cos_half_theta = cos(theta / 2);

				${c.complex.name} minus_i = ${c.complex.ctr}(0, -1);

				${c.complex.name} k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(${c.complex.ctr}(0, -phi))
				), sin_half_theta);

				${c.complex.name} k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(${c.complex.ctr}(0, phi))
				), sin_half_theta);

				a[index] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				b[index] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}

			void propagationFunc(${c.complex.name} *a_res, ${c.complex.name} *b_res,
				${c.complex.name} a, ${c.complex.name} b,
				${c.complex.name} ka, ${c.complex.name} kb,
				${c.scalar.name} t, ${c.scalar.name} dt,
				${c.scalar.name} kvector, ${c.scalar.name} potential,
				${c.scalar.name} phi)
			{
				${c.scalar.name} n_a = squared_abs(a);
				${c.scalar.name} n_b = squared_abs(b);

				${c.complex.name} ta = complex_mul_scalar(ka, kvector) - complex_mul_scalar(a, potential);
				${c.complex.name} tb = complex_mul_scalar(kb, kvector) - complex_mul_scalar(b, potential);

				${c.scalar.name} phase = t * (${c.scalar.name})${detuning} + phi;
				${c.scalar.name} sin_phase = ${c.rabi_freq / 2} * sin(phase);
				${c.scalar.name} cos_phase = ${c.rabi_freq / 2} * cos(phase);

				*a_res = ${c.complex.ctr}(-ta.y, ta.x) +
					complex_mul(${c.complex.ctr}(
						- n_a * n_a * ${c.l111 / 2} - n_b * ${c.l12 / 2},
						- n_a * ${c.g11} - n_b * ${c.g12}), a) -
					complex_mul(${c.complex.ctr}(sin_phase, cos_phase), b);

				*b_res = ${c.complex.ctr}(-tb.y, tb.x) +
					complex_mul(${c.complex.ctr}(
						- n_a * ${c.l12 / 2} - n_b * ${c.l22 / 2},
						- n_a * ${c.g12} - n_b * ${c.g22}), b) -
					complex_mul(${c.complex.ctr}(-sin_phase, cos_phase), a);
			}

			__kernel void calculateRK(__global ${c.complex.name} *a, __global ${c.complex.name} *b,
				__global ${c.complex.name} *a_copy, __global ${c.complex.name} *b_copy,
				__global ${c.complex.name} *a_kdata, __global ${c.complex.name} *b_kdata,
				__global ${c.complex.name} *a_res, __global ${c.complex.name} *b_res,
				${c.scalar.name} t, ${c.scalar.name} dt,
				read_only image3d_t potentials, read_only image3d_t kvectors,
				${c.scalar.name} phi, int stage)
			{
				DEFINE_INDEXES;

				${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k % ${c.nvz});
				${c.scalar.name} potential = get_float_from_image(potentials, i, j, k % ${c.nvz});

				${c.complex.name} ra = a_res[index];
				${c.complex.name} rb = b_res[index];
				${c.complex.name} ka = a_kdata[index];
				${c.complex.name} kb = b_kdata[index];
				${c.complex.name} a0 = a_copy[index];
				${c.complex.name} b0 = b_copy[index];

				${c.scalar.name} val_coeffs[4] = {0.5, 0.5, 1.0};
				${c.scalar.name} res_coeffs[4] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

				${c.complex.name} a_val, b_val;
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

				a[index] += complex_mul_scalar(ra, dt * res_coeffs[stage]);
				b[index] += complex_mul_scalar(rb, dt * res_coeffs[stage]);
			}
		"""

		self._program = self._env.compile(kernels, self._constants, detuning=self._detuning)
		self._calculateMatrix = self._program.calculateMatrix
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

		a[:,:,:] = a0 * k1 + b0 * k2
		b[:,:,:] = a0 * k3 + b0 * k1

	def _gpu__applyMatrix(self, cloud, theta, phi):
		self._calculateMatrix(cloud.a.shape, cloud.a.data, cloud.b.data,
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

		shape = cloud.a.shape
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
			func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(0))

			fft(a_res, a_kdata, inverse=True, batch=batch)
			fft(b_res, b_kdata, inverse=True, batch=batch)
			func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(1))

			fft(a_res, a_kdata, inverse=True, batch=batch)
			fft(b_res, b_kdata, inverse=True, batch=batch)
			func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(2))

			fft(a_res, a_kdata, inverse=True, batch=batch)
			fft(b_res, b_kdata, inverse=True, batch=batch)
			func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
				a_res, b_res, t, dt, p, k, phi, numpy.int32(3))

	def _propagationFunc(self, a_data, b_data, a_kdata, b_kdata, a_res, b_res, t, dt, phi):

		batch = a_data.size / self._constants.cells
		nvz = self._constants.nvz

		# TODO: remove hardcoding (g must depend on cloud.a.comp and cloud.b.comp)
		g11 = self._constants.g11
		g12 = self._constants.g12
		g22 = self._constants.g22

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
			1j * (n_a * g11 + n_b * g12)) * a_data - \
			0.5j * self._constants.rabi_freq * \
				numpy.exp(1j * (- t * self._detuning - phi)) * b_data

		b_res += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22 + n_a * g12)) * b_data - \
			0.5j * self._constants.rabi_freq * \
				numpy.exp(1j * (t * self._detuning + phi)) * a_data

	def apply(self, cloud, theta, matrix=True):
		phi =  cloud.time * self._detuning + self._starting_phase
		t_pulse = theta * self._constants.rabi_period

		if matrix:
			self._applyMatrix(cloud, theta, phi)
		else:
			self._applyReal(cloud, t_pulse, phi)

		cloud.time += t_pulse


class SplitStepEvolution(PairedCalculation):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

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

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			__kernel void propagateKSpaceRealTime(__global ${c.complex.name} *a, __global ${c.complex.name} *b,
				${c.scalar.name} dt, read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k % ${c.nvz});
				${c.scalar.name} prop_angle = kvector * dt / 2;
				${c.complex.name} prop_coeff = ${c.complex.ctr}(native_cos(prop_angle), native_sin(prop_angle));

				${c.complex.name} a0 = a[index];
				${c.complex.name} b0 = b[index];

				a[index] = complex_mul(a0, prop_coeff);
				b[index] = complex_mul(b0, prop_coeff);
			}

			// Propagates state vector in x-space for evolution calculation
			%for suffix in ('', 'Wigner'):
			__kernel void propagateXSpaceTwoComponent${suffix}(__global ${c.complex.name} *aa,
				__global ${c.complex.name} *bb, ${c.scalar.name} dt,
				read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				${c.scalar.name} V = get_float_from_image(potentials, i, j, k % ${c.nvz});

				${c.complex.name} a = aa[index];
				${c.complex.name} b = bb[index];

				//store initial x-space field
				${c.complex.name} a0 = a;
				${c.complex.name} b0 = b;

				${c.complex.name} pa, pb, da = ${c.complex.ctr}(0, 0), db = ${c.complex.ctr}(0, 0);
				${c.scalar.name} n_a, n_b;

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					n_a = squared_abs(a);
					n_b = squared_abs(b);

					// TODO: there must be no minus sign before imaginary part,
					// but without it the whole thing diverges
					pa = ${c.complex.ctr}(
						-(${c.l111} * n_a * n_a + ${c.l12} * n_b) / 2,
						-(-V - ${c.g11} * n_a - ${c.g12} * n_b));
					pb = ${c.complex.ctr}(
						-(${c.l22} * n_b + ${c.l12} * n_a) / 2,
						-(-V - ${c.g22} * n_b - ${c.g12} * n_a));

					%if suffix == "Wigner":
						pa += ${c.complex.ctr}(
							(1.5 * n_a - 0.75 / ${c.dV}) * ${c.l111} + ${c.l12} * 0.25,
							-(${c.g11} + 0.5 * ${c.g12})) / ${c.dV};
						pb += ${c.complex.ctr}(
							${c.l12} * 0.25 + ${c.l22} * 0.5,
							-(${c.g22} + 0.5 * ${c.g12})) / ${c.dV};
					%endif

					// calculate midpoint log derivative and exponentiate
					da = cexp(complex_mul_scalar(pa, (dt / 2)));
					db = cexp(complex_mul_scalar(pb, (dt / 2)));

					//propagate to midpoint using log derivative
					a = complex_mul(a0, da);
					b = complex_mul(b0, db);
				%endfor

				//propagate to endpoint using log derivative
				aa[index] = complex_mul(a, da);
				bb[index] = complex_mul(b, db);
			}
			%endfor
		"""

		self._program = self._env.compile(kernels, self._constants)
		self._kpropagate_func = self._program.propagateKSpaceRealTime
		self._xpropagate_func = self._program.propagateXSpaceTwoComponent
		self._xpropagate_wigner = self._program.propagateXSpaceTwoComponentWigner

	def _toKSpace(self, cloud):
		batch = cloud.a.size / self._constants.cells
		self._plan.execute(cloud.a.data, batch=batch, inverse=True)
		self._plan.execute(cloud.b.data, batch=batch, inverse=True)

	def _toXSpace(self, cloud):
		batch = cloud.a.size / self._constants.cells
		self._plan.execute(cloud.a.data, batch=batch)
		self._plan.execute(cloud.b.data, batch=batch)

	def _gpu__kpropagate(self, cloud, dt):
		self._kpropagate_func(cloud.a.shape,
			cloud.a.data, cloud.b.data, self._constants.scalar.cast(dt), self._kvectors)

	def _cpu__kpropagate(self, cloud, dt):
		kcoeff = numpy.exp(self._kvectors * (1j * dt / 2))
		data1 = cloud.a.data
		data2 = cloud.b.data
		nvz = self._constants.nvz

		for e in xrange(cloud.a.size / self._constants.cells):
			start = e * nvz
			stop = (e + 1) * nvz
			data1[start:stop,:,:] *= kcoeff
			data2[start:stop,:,:] *= kcoeff

	def _gpu__xpropagate(self, cloud, dt):
		if cloud.type == WIGNER:
			func = self._xpropagate_wigner
		else:
			func = self._xpropagate_func

		func(cloud.a.shape, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(dt), self._potentials)

	def _cpu__getNoiseTerms(self, cloud):
		noise_a = numpy.zeros(self._constants.ens_shape, dtype=self._constants.complex.dtype)
		noise_b = numpy.zeros(self._constants.ens_shape, dtype=self._constants.complex.dtype)

		shape = self._constants.ens_shape

		eta = [numpy.random.normal(scale=math.sqrt(self._constants.dt_evo / self._constants.dV),
			size=shape).astype(self._constants.scalar.dtype) for i in xrange(4)]

		n1 = numpy.abs(cloud.a.data) ** 2
		n2 = numpy.abs(cloud.b.data) ** 2
		dV = self._constants.dV
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = 0.25 * l12 * (n2 - 0.5 / dV) + 0.25 * l111 * (3.0 * n1 * n1 - 6.0 * n1 / dV + 1.5 / dV / dV)
		d = 0.25 * l12 * (n1 - 0.5 / dV) + 0.25 * l22 * (2.0 * n2 - 1.0 / dV)
		t = 0.25 * l12 * cloud.a.data * numpy.conj(cloud.b.data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		row1 = t1 * eta[0]
		row2 = b / t1 * eta[0] + t2 * eta[1]
		row3 = c / t2 * eta[1] + t3 * eta[2]
		row4 = -c / t1 * eta[0] + b * c / (a * t2) * eta[1] + b / a * t3 * eta[2] + \
			numpy.sqrt((a ** 2 - b ** 2 - c ** 2) / a) * eta[3]

		noise_a = row1 + 1j * row3
		noise_b = row2 + 1j * row4

		noise_a /= (cloud.a.data * self._constants.dt_evo)
		noise_b /= (cloud.b.data * self._constants.dt_evo)

		#print numpy.sum(numpy.abs(noise_a))
		#print numpy.sum(numpy.abs(numpy.nan_to_num(noise_a)))

		return numpy.nan_to_num(noise_a), numpy.nan_to_num(noise_b)

	def _cpu__xpropagate(self, cloud, dt):
		a = cloud.a
		b = cloud.b
		a0 = a.data.copy()
		b0 = b.data.copy()

		comp1 = cloud.a.comp
		comp2 = cloud.b.comp
		g = self._constants.g
		g11 = g[(comp1, comp1)]
		g12 = g[(comp1, comp2)]
		g22 = g[(comp2, comp2)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		p = self._potentials * 1j
		nvz = self._constants.nvz

		if cloud.type == WIGNER:
			noise_a, noise_b = self._getNoiseTerms(cloud)

		for iter in xrange(self._constants.itmax):
			n_a = numpy.abs(a.data) ** 2
			n_b = numpy.abs(b.data) ** 2

			pa = n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) + \
				1j * (n_a * g11 + n_b * g12)

			pb = n_b * (-l22 / 2) + n_a * (-l12 / 2) + \
				1j * (n_b * g22 + n_a * g12)

			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				pa[start:stop] += p
				pb[start:stop] += p

			if cloud.type == WIGNER:
				dV = self._constants.dV
				pa += ((1.5 * n_a - 0.75 / dV) * l111 + l12 * 0.25 -
					1j * (g11 + 0.5 * g12)) / dV
				pb += (l12 * 0.25 + l22 * 0.5 -
					1j * (g22 + 0.5 * g12)) / dV

				#print numpy.sum(numpy.abs(noise_a)) / numpy.sum(numpy.abs(pa)), \
				#	numpy.sum(numpy.abs(noise_b)) / numpy.sum(numpy.abs(pb))

				pa += noise_a * 1j
				pb += noise_b * 1j

			da = numpy.exp(pa * (dt / 2))
			db = numpy.exp(pb * (dt / 2))

			a.data = a0 * da
			b.data = b0 * db

		a.data *= da
		b.data *= db

	def _finishStep(self, cloud, dt):
		if self._midstep:
			self._kpropagate(cloud, dt)
			self._midstep = False

	def propagate(self, cloud, dt):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them
		if self._midstep:
			self._kpropagate(cloud, dt * 2)
		else:
			self._kpropagate(cloud, dt)

		self._toXSpace(cloud)
		self._xpropagate(cloud, dt)
		cloud.time += dt

		self._midstep = True
		self._toKSpace(cloud)

	def _runCallbacks(self, t, cloud, callbacks):
		if callbacks is None:
			return

		self._finishStep(cloud, self._constants.dt_evo)
		self._toXSpace(cloud)
		for callback in callbacks:
			callback(t, cloud)
		self._toKSpace(cloud)

	def run(self, cloud, time, callbacks=None, callback_dt=0):

		# in SI units
		t = 0
		callback_t = 0
		t_rho = self._constants.t_rho

		# in natural units
		dt = self._constants.dt_evo

		self._toKSpace(cloud)

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:
				self.propagate(cloud, dt)
				t += dt * t_rho
				callback_t += dt * t_rho

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

			self._toXSpace(cloud)

		except TerminateEvolution:
			return t


class RungeKuttaEvolution:

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _propagationFunc(self, a_data, b_data, dt):

		batch = a_data.size / self._constants.cells
		nvz = self._constants.nvz

		# TODO: remove hardcoding (g must depend on cloud.a.comp and cloud.b.comp)
		g11 = self._constants.g11
		g12 = self._constants.g12
		g22 = self._constants.g22

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		n_a = numpy.abs(a_data) ** 2
		n_b = numpy.abs(b_data) ** 2

		a_res = self._env.allocate(a_data.shape, dtype=a_data.dtype)
		b_res = self._env.allocate(b_data.shape, dtype=b_data.dtype)
		a_kdata = self._env.allocate(a_data.shape, dtype=a_data.dtype)
		b_kdata = self._env.allocate(b_data.shape, dtype=b_data.dtype)

		self._plan.execute(a_data, a_kdata, inverse=True, batch=batch)
		self._plan.execute(b_data, b_kdata, inverse=True, batch=batch)

		for e in xrange(batch):
			start = e * nvz
			stop = (e + 1) * nvz
			a_res[start:stop,:,:] = 1j * (a_kdata[start:stop,:,:] * self._kvectors -
				a_data[start:stop,:,:] * self._potentials)
			b_res[start:stop,:,:] = 1j * (b_kdata[start:stop,:,:] * self._kvectors -
				b_data[start:stop,:,:] * self._potentials)

		a_res += (n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11 + n_b * g12)) * a_data

		b_res += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22 + n_a * g12)) * b_data

		return a_res, b_res

	def _noiseFunc(self, a_data, b_data):
		shape = self._constants.ens_shape

		noise_a = numpy.zeros(shape, dtype=self._constants.complex.dtype)
		noise_b = numpy.zeros(shape, dtype=self._constants.complex.dtype)

		eta = [numpy.random.normal(scale=math.sqrt(self._constants.dt_evo / self._constants.dV),
			size=shape).astype(self._constants.scalar.dtype) for i in xrange(4)]

		n1 = numpy.abs(cloud.a.data) ** 2
		n2 = numpy.abs(cloud.b.data) ** 2
		dV = self._constants.dV
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = 0.25 * l12 * n2 + 0.25 * l111 * 3.0 * n1 * n1
		d = 0.25 * l12 * n1 + 0.25 * l22 * 2.0 * n2
		t = 0.25 * l12 * cloud.a.data * numpy.conj(cloud.b.data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		row1 = t1 * eta[0]
		row2 = b / t1 * eta[0] + t2 * eta[1]
		row3 = c / t2 * eta[1] + t3 * eta[2]
		row4 = -c / t1 * eta[0] + b * c / (a * t2) * eta[1] + b / a * t3 * eta[2] + \
			numpy.sqrt((a ** 2 - b ** 2 - c ** 2) / a) * eta[3]

		noise_a = row1 + 1j * row3
		noise_b = row2 + 1j * row4

		return numpy.nan_to_num(noise_a), numpy.nan_to_num(noise_b)

	def propagate(self, cloud, dt):

		main_a, main_b = self._propagationFunc(cloud.a.data, cloud.b.data, dt)
		noise_a, noise_b = self._noiseFunc(cloud.a.data, cloud.b.data)

		temp_a = cloud.a.data + main_a * dt + noise_a * math.sqrt(dt)
		temp_b = cloud.b.data + main_b * dt + noise_b * math.sqrt(dt)

		noise_mod_a, noise_mod_b = self._noiseFunc(temp_a, temp_b)

		cloud.a.data += main_a * dt + noise_a * deltaW_a + \
			0.5 * (noise_mod_a - noise_a) * (deltaW_a ** 2 - dt) * math.sqrt(dt)
		cloud.b.data += main_b * dt + noise_b * deltaW_b + \
			0.5 * (noise_mod_b - noise_b) * (deltaW_b ** 2 - dt) * math.sqrt(dt)

		cloud.time += dt

	def _runCallbacks(self, t, cloud, callbacks):
		if callbacks is None:
			return

		for callback in callbacks:
			callback(t, cloud)

	def run(self, cloud, time, callbacks=None, callback_dt=0):

		# in SI units
		t = 0
		callback_t = 0
		t_rho = self._constants.t_rho

		# in natural units
		dt = self._constants.dt_evo

		self._toKSpace(cloud)

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:
				self.propagate(cloud, dt)
				t += dt * t_rho
				callback_t += dt * t_rho

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

			self._toXSpace(cloud)

		except TerminateEvolution:
			return t
