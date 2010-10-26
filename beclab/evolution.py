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


class SplitStepEvolution(PairedCalculation):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)
		self._random = createRandom(env, constants.double)

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._projector_mask, _ = getProjectorMask(self._env, self._constants)

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
				from math import sqrt
			%>

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			EXPORTED_FUNC void propagateKSpaceRealTime(GLOBAL_MEM COMPLEX *a, GLOBAL_MEM COMPLEX *b,
				SCALAR dt, GLOBAL_MEM SCALAR *kvectors)
			{
				DEFINE_INDEXES;

				SCALAR kvector = kvectors[cell_index];
				SCALAR prop_angle = kvector * dt / 2;
				COMPLEX prop_coeff = complex_ctr(cos(prop_angle), sin(prop_angle));

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				a[index] = complex_mul(a0, prop_coeff);
				b[index] = complex_mul(b0, prop_coeff);
			}

			// Propagates state vector in x-space for evolution calculation
			EXPORTED_FUNC void propagateXSpaceTwoComponent(GLOBAL_MEM COMPLEX *aa,
				GLOBAL_MEM COMPLEX *bb, SCALAR dt, GLOBAL_MEM SCALAR *potentials)
			{
				DEFINE_INDEXES;

				SCALAR V = potentials[cell_index];

				COMPLEX a = aa[index];
				COMPLEX b = bb[index];

				//store initial x-space field
				COMPLEX a0 = a;
				COMPLEX b0 = b;

				COMPLEX pa, pb, da = complex_ctr(0, 0), db = complex_ctr(0, 0);
				SCALAR n_a, n_b;

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

					// TODO: there must be no minus sign before imaginary part,
					// but without it the whole thing diverges
					pa = complex_ctr(
						-(temp * temp * ${c.l111 * 1e20} + ${c.l12} * n_b) / 2,
						-(-V - ${g11} * n_a - ${g12} * n_b));
					pb = complex_ctr(
						-(${c.l22} * n_b + ${c.l12} * n_a) / 2,
						-(-V - ${g22} * n_b - ${g12} * n_a));

					/*
					pa += complex_ctr(
						-(${1.5 * c.l111 / c.V / c.V} - ${3.0 * c.l111 / c.V * 1e10} * temp -
						${0.5 * c.l12 / c.V}) / 2,
						-(${g11 / c.V} + ${0.5 * g12 / c.V})
					);

					pb += complex_ctr(
						-(-${c.l22 / c.V} - ${0.5 * c.l12 / c.V}) / 2,
						-(${g22 / c.V} + ${0.5 * g12 / c.V})
					);*/

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

			// Propagates state vector in x-space for evolution calculation
			EXPORTED_FUNC void addNoise(GLOBAL_MEM COMPLEX *a,
				GLOBAL_MEM COMPLEX *b, SCALAR dt,
				GLOBAL_MEM COMPLEX *randoms)
			{
				DEFINE_INDEXES;

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				COMPLEX r_a = randoms[index];
				COMPLEX r_b = randoms[index + ${c.cells * c.ensembles}];

				SCALAR n_a = squared_abs(a0);
				SCALAR n_b = squared_abs(b0);

				SCALAR st = sqrt(dt / (SCALAR)${c.dV * 2});

				// FIXME: Some magic here. l111 ~ 10^-42, while single precision float
				// can only handle 10^-38.
				SCALAR t_a = n_a * ${1.0e-10};

				SCALAR d11 = sqrt(t_a * t_a * (SCALAR)${9.0 * c.l111 * 1e20} +
					(SCALAR)${c.l12} * n_b) * st;
				SCALAR d22 = sqrt((SCALAR)${c.l12} * n_a +
					(SCALAR)${4.0 * c.l22} * n_b) * st;

				a[index] = a0 + complex_mul_scalar(r_a, d11);
				b[index] = b0 + complex_mul_scalar(r_b, d22);
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
			COMP_1_minus1=COMP_1_minus1, COMP_2_1=COMP_2_1)
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
		kcoeff = numpy.exp(self._kvectors * (1j * dt / 2))
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

	def _gpu__xpropagate(self, cloud, dt):
		self._xpropagate_func(cloud.a.size, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(dt), self._potentials)

	def _cpu__xpropagate(self, cloud, dt):
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

		p = self._potentials * 1j
		nvz = self._constants.nvz

		for iter in xrange(self._constants.itmax):
			n_a = numpy.abs(a.data) ** 2
			n_b = numpy.abs(b.data) ** 2

			pa = n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) + \
				1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)

			pb = n_b * (-l22 / 2) + n_a * (-l12 / 2) + \
				1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)

			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				pa[start:stop] += p
				pb[start:stop] += p

			da = numpy.exp(pa * (dt / 2))
			db = numpy.exp(pb * (dt / 2))

			a.data = a0 * da
			b.data = b0 * db

		a.data *= da
		b.data *= db

	def _noiseFunc(self, a_data, b_data, dt):
		coeff = math.sqrt(1.0 / self._constants.dV)

		n1 = numpy.abs(a_data) ** 2
		n2 = numpy.abs(b_data) ** 2
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = l12 * n2 + (9.0 * l111) * n1 * n1
		d = l12 * n1 + (4.0 * l22) * n2
		t = l12 * a_data * numpy.conj(b_data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		k1 = numpy.nan_to_num(c / t2)
		k2 = numpy.nan_to_num(t3)
		k3 = numpy.nan_to_num(b / t1)
		k4 = numpy.nan_to_num(t2)
		k5 = numpy.nan_to_num(-c / t1)
		k6 = numpy.nan_to_num(b * c / (a * t2))
		k7 = numpy.nan_to_num(b / a * t3)
		k8 = numpy.nan_to_num(numpy.sqrt((d * a - b ** 2 - c ** 2) / a))

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)

		return \
			coeff * t1 + 1j * zeros, \
			zeros + 1j * coeff * k1, \
			zeros + 1j * coeff * k2, \
			zeros + 1j * zeros, \
			coeff * (k3 + 1j * k5), \
			coeff * (k4 + 1j * k6), \
			coeff * (zeros + 1j * k7), \
			coeff * (zeros + 1j * k8)

	def _noiseFunc2(self, a_data, b_data, dt):

		n1 = numpy.abs(a_data) ** 2
		n2 = numpy.abs(b_data) ** 2
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		st = math.sqrt(1.0 / self._constants.dV)
		d11 = numpy.sqrt(9.0 * l111 * (n1 ** 2) + l12 * n2) * st
		d22 = numpy.sqrt(l12 * n1 + 4.0 * l22 * n2) * st

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)
		czeros = zeros + 1j * zeros

		return \
			d11 + 1j * zeros, \
			czeros, \
			zeros + 1j * d11, \
			czeros, \
			czeros, \
			d22 + 1j * zeros, \
			czeros, \
			zeros + 1j * d22

	def _propagateNoise(self, cloud, dt):

		shape = self._constants.ens_shape
		Z1 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]
		Z0 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]

		G101, G102, G103, G104, G201, G202, G203, G204 = self._noiseFunc2(cloud.a.data, cloud.b.data, dt)
		G111, G112, G113, G114, G211, G212, G213, G214 = \
			self._noiseFunc2(
				cloud.a.data + math.sqrt(dt / 2) * (
					G101 * Z1[0] + G102 * Z1[1] + G103 * Z1[2] + G104 * Z1[3]),
				cloud.b.data + math.sqrt(dt / 2) * (
					G201 * Z1[0] + G202 * Z1[1] + G203 * Z1[2] + G204 * Z1[3]), dt)

		G121, G122, G123, G124, G221, G222, G223, G224 = \
			self._noiseFunc2(
				cloud.a.data + math.sqrt(dt / 2) * (
					G101 * Z1[0] + G102 * Z1[1] + G103 * Z1[2] + G104 * Z1[3]),
				cloud.b.data + math.sqrt(dt / 2) * (
					G201 * Z1[0] + G202 * Z1[1] + G203 * Z1[2] + G204 * Z1[3]), dt)

		cloud.a.data += 0.5 * math.sqrt(dt) * (
				(G111 + G121) * Z0[0] +
				(G112 + G122) * Z0[1] +
				(G113 + G123) * Z0[2] +
				(G114 + G124) * Z0[3]
			)

		cloud.b.data += 0.5 * math.sqrt(dt) * (
				(G211 + G221) * Z0[0] +
				(G212 + G222) * Z0[1] +
				(G213 + G223) * Z0[2] +
				(G214 + G224) * Z0[3]
			)

	def _cpu__propagateNoise2(self, cloud, dt):
		shape = self._constants.ens_shape
		Z0 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]

		n1 = numpy.abs(cloud.a.data) ** 2
		n2 = numpy.abs(cloud.b.data) ** 2
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		st = math.sqrt(dt / self._constants.dV)
		d11 = numpy.sqrt(9.0 * l111 * (n1 ** 2) + l12 * n2) * st
		d22 = numpy.sqrt(l12 * n1 + 4.0 * l22 * n2) * st

		cloud.a.data += d11 * (Z0[0] + 1j * Z0[1])
		cloud.b.data += d22 * (Z0[2] + 1j * Z0[3])

	def _gpu__propagateNoise2(self, cloud, dt):
		randoms = self._random.random_normal(size=cloud.a.size * 2)

		self._addnoise_func(cloud.a.size, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(dt), randoms)

	def _finishStep(self, cloud, dt):
		if self._midstep:
			self._kpropagate(cloud, dt)
			self._midstep = False

	def propagate(self, cloud, dt, noise):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them
		if self._midstep:
			self._kpropagate(cloud, dt * 2)
		else:
			self._kpropagate(cloud, dt)

		if cloud.type == WIGNER and noise:
			self._projector(cloud)

		self._toXSpace(cloud)
		self._xpropagate(cloud, dt)

		if cloud.type == WIGNER and noise:
			self._propagateNoise2(cloud, dt)

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

	def run(self, cloud, time, callbacks=None, callback_dt=0, noise=True):

		t = 0
		callback_t = 0

		# in natural units
		dt = self._constants.dt_evo

		self._toKSpace(cloud)

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:
				self.propagate(cloud, dt, noise)
				t += dt
				callback_t += dt

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

			self._toXSpace(cloud)

		except TerminateEvolution:
			return t


class RungeKuttaEvolution(PairedCalculation):

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _driftFunc(self, a_data, b_data):

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

		# needs to be wrapped in projection
		a_res += (n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11 + n_b * g12)) * a_data

		# needs to be wrapped in projection
		b_res += (n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22 + n_a * g12)) * b_data

		return a_res, b_res

	def _noiseFunc(self, a_data, b_data, dt):
		coeff = 0 # math.sqrt(dt / self._constants.dV)

		n1 = numpy.abs(a_data) ** 2
		n2 = numpy.abs(b_data) ** 2
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = 0.125 * l12 * n2 + 0.375 * l111 * n1 * n1
		d = 0.125 * l12 * n1 + 0.25 * l22 * n2
		t = 0.25 * l12 * a_data * numpy.conj(b_data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		k1 = numpy.nan_to_num(c / t2)
		k2 = numpy.nan_to_num(t3)
		k3 = numpy.nan_to_num(b / t1)
		k4 = numpy.nan_to_num(t2)
		k5 = numpy.nan_to_num(-c / t1)
		k6 = numpy.nan_to_num(b * c / (a * t2))
		k7 = numpy.nan_to_num(b / a * t3)
		k8 = numpy.nan_to_num(numpy.sqrt((a ** 2 - b ** 2 - c ** 2) / a))

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)

		return \
			coeff * t1 + 1j * zeros, \
			zeros + 1j * coeff * k1, \
			zeros + 1j * coeff * k2, \
			zeros + 1j * zeros, \
			coeff * (k3 + 1j * k5), \
			coeff * (k4 + 1j * k6), \
			coeff * (zeros + 1j * k7), \
			coeff * (zeros + 1j * k8)

	def propagate(self, cloud, dt):

		print numpy.sum(numpy.abs(cloud.a.data)), numpy.sum(numpy.abs(cloud.b.data))

		shape = self._constants.ens_shape
		Z1 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]
		Z0 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]

		Omega11, Omega21 = self._driftFunc(cloud.a.data, cloud.b.data)
		G101, G102, G103, G104, G201, G202, G203, G204 = self._noiseFunc(cloud.a.data, cloud.b.data, dt)
		G111, G112, G113, G114, G211, G212, G213, G214 = \
			self._noiseFunc(
				cloud.a.data + math.sqrt(dt / 2) * (
					G101 * Z1[0] + G102 * Z1[1] + G103 * Z1[2] + G104 * Z1[3]),
				cloud.b.data + math.sqrt(dt / 2) * (
					G201 * Z1[0] + G202 * Z1[1] + G203 * Z1[2] + G204 * Z1[3]), dt)

		G121, G122, G123, G124, G221, G222, G223, G224 = \
			self._noiseFunc(
				cloud.a.data + dt * Omega11 + math.sqrt(dt / 2) * (
					G101 * Z1[0] + G102 * Z1[1] + G103 * Z1[2] + G104 * Z1[3]),
				cloud.b.data + dt * Omega21 + math.sqrt(dt / 2) * (
					G201 * Z1[0] + G202 * Z1[1] + G203 * Z1[2] + G204 * Z1[3]), dt)

		Omega12, Omega22 = self._driftFunc(
				cloud.a.data + dt * Omega11 + math.sqrt(dt) * (
					G101 * Z0[0] + G102 * Z0[1] + G103 * Z0[2] + G104 * Z0[3]),
				cloud.b.data + dt * Omega21 + math.sqrt(dt) * (
					G201 * Z0[0] + G202 * Z0[1] + G203 * Z0[2] + G204 * Z0[3]))

		cloud.a.data += 0.5 * dt * (Omega11 + Omega12) + \
			0.5 * math.sqrt(dt) * (
				(G111 + G121) * Z0[0] +
				(G112 + G122) * Z0[1] +
				(G113 + G123) * Z0[2] +
				(G114 + G124) * Z0[3]
			)

		cloud.b.data += 0.5 * dt * (Omega21 + Omega22) + \
			0.5 * math.sqrt(dt) * (
				(G211 + G221) * Z0[0] +
				(G212 + G222) * Z0[1] +
				(G213 + G223) * Z0[2] +
				(G214 + G224) * Z0[3]
			)

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

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:
				print t
				self.propagate(cloud, dt)
				t += dt * t_rho
				callback_t += dt * t_rho

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

		except TerminateEvolution:
			return t


class RK4Evolution(PairedCalculation):

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)
		self._projector_mask, _ = getProjectorMask(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _cpu__calculateRK(self, _, a_data, b_data, a_copy, b_copy, a_kdata, b_kdata,
			a_res, b_res, t, dt, p, k, stage, randoms):

		val_coeffs = (0.5, 0.5, 1.0, 0.0)
		res_coeffs = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)

		if stage == 0:
			a = a_data.copy()
			b = b_data.copy()
		else:
			a = a_res.copy()
			b = b_res.copy()

		self._propagationFunc(a, b, a_kdata, b_kdata, a_res, b_res, t, dt, randoms)

		a_data += a_res * (dt * res_coeffs[stage])
		b_data += b_res * (dt * res_coeffs[stage])

		a_res[:,:,:] = a_copy + a_res * (dt * val_coeffs[stage])
		b_res[:,:,:] = b_copy + b_res * (dt * val_coeffs[stage])

	def _noiseFunc(self, a_data, b_data, dt, randoms):
		coeff = math.sqrt(dt / self._constants.dV)

		n1 = numpy.abs(a_data) ** 2
		n2 = numpy.abs(b_data) ** 2
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = 0.125 * l12 * n2 + 0.375 * l111 * n1 * n1
		d = 0.125 * l12 * n1 + 0.25 * l22 * n2
		t = 0.25 * l12 * a_data * numpy.conj(b_data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		k1 = numpy.nan_to_num(c / t2)
		k2 = numpy.nan_to_num(t3)
		k3 = numpy.nan_to_num(b / t1)
		k4 = numpy.nan_to_num(t2)
		k5 = numpy.nan_to_num(-c / t1)
		k6 = numpy.nan_to_num(b * c / (a * t2))
		k7 = numpy.nan_to_num(b / a * t3)
		k8 = numpy.nan_to_num(numpy.sqrt((a ** 2 - b ** 2 - c ** 2) / a))

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)

		return \
			coeff * randoms[0] * t1 + \
			1j * coeff * randoms[1] * k1 + \
			1j * coeff * randoms[2] * k2, \
			coeff * randoms[0] * k3 + 1j * coeff * randoms[0] * k5 + \
			coeff * randoms[1] * k4 + 1j * coeff * randoms[1] * k6 + \
			1j * coeff * randoms[2] * k7 + \
			1j * coeff * randoms[3] * k8

	def _projector(self, data):
		shape = data.shape
		dtype = data.dtype
		batch = data.size / self._constants.cells

		kdata = self._env.allocate(shape, dtype=dtype)
		res = self._env.allocate(shape, dtype=dtype)

		self._plan.execute(data, kdata, inverse=True, batch=batch)

		for e in xrange(batch):
			start = e * nvz
			stop = (e + 1) * nvz
			kdata[start:stop,:,:] *= self._projector_mask

		self._plan.execute(kdata, res, batch=batch)
		return res

	def _propagationFunc(self, a_data, b_data, a_kdata, b_kdata, a_res, b_res, t, dt, randoms):

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

		a_noise, b_noise = self._noiseFunc(a_data, b_data, dt, randoms)

		a_res += self._projector((n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) -
			1j * (n_a * g11 + n_b * g12)) * a_data + a_noise)

		b_res += self._projector((n_b * (-l22 / 2) + n_a * (-l12 / 2) -
			1j * (n_b * g22 + n_a * g12)) * b_data + b_noise)

	def _runCallbacks(self, t, cloud, callbacks):
		if callbacks is None:
			return

		for callback in callbacks:
			callback(t, cloud)

	def run(self, cloud, time, callbacks=None, callback_dt=0):

		batch = cloud.a.size / self._constants.cells
		shape = cloud.a.shape

		func = self._calculateRK
		fft = self._plan.execute
		cast = self._constants.scalar.cast
		p = self._potentials
		k = self._kvectors

		# in natural units
		dt = cast(self._constants.dt_evo)

		shape = cloud.a.shape
		dtype = cloud.a.dtype

		a_copy = self._env.allocate(shape, dtype=dtype)
		b_copy = self._env.allocate(shape, dtype=dtype)
		a_kdata = self._env.allocate(shape, dtype=dtype)
		b_kdata = self._env.allocate(shape, dtype=dtype)
		a_res = self._env.allocate(shape, dtype=dtype)
		b_res = self._env.allocate(shape, dtype=dtype)

		# in SI units
		t = 0
		callback_t = 0
		t_rho = self._constants.t_rho

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:

				t_nat = t / t_rho

				self._env.copyBuffer(cloud.a.data, a_copy)
				self._env.copyBuffer(cloud.b.data, b_copy)

				randoms = [numpy.random.normal(scale=1, size=shape).astype(
					self._constants.scalar.dtype) for i in xrange(4)]

				fft(a_copy, a_kdata, inverse=True, batch=batch)
				fft(b_copy, b_kdata, inverse=True, batch=batch)
				func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
					a_res, b_res, t_nat, dt, p, k, numpy.int32(0), randoms)

				fft(a_res, a_kdata, inverse=True, batch=batch)
				fft(b_res, b_kdata, inverse=True, batch=batch)
				func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
					a_res, b_res, t_nat, dt, p, k, numpy.int32(1), randoms)

				fft(a_res, a_kdata, inverse=True, batch=batch)
				fft(b_res, b_kdata, inverse=True, batch=batch)
				func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
					a_res, b_res, t_nat, dt, p, k, numpy.int32(2), randoms)

				fft(a_res, a_kdata, inverse=True, batch=batch)
				fft(b_res, b_kdata, inverse=True, batch=batch)
				func(shape, cloud.a.data, cloud.b.data, a_copy, b_copy, a_kdata, b_kdata,
					a_res, b_res, t_nat, dt, p, k, numpy.int32(3), randoms)

				t += dt * t_rho
				callback_t += dt * t_rho

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

		except TerminateEvolution:
			return t


class SplitStepEvolution2(PairedCalculation):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createFFTPlan(env, constants.shape, constants.complex.dtype)

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._projector_mask, _ = getProjectorMask(self._env, self._constants)

		self._prepare()

	def _projector(self, data):
		nvz = self._constants.nvz
		for e in xrange(self._constants.ensembles):
			start = e * nvz
			stop = (e + 1) * nvz
			if self._constants.dim == 3:
				data[start:stop,:,:] *= self._projector_mask
			else:
				data[start:stop] *= self._projector_mask

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):

		kernels = """
			<%!
				from math import sqrt
			%>

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			EXPORTED_FUNC void propagateKSpaceRealTime(GLOBAL_MEM COMPLEX *a, GLOBAL_MEM COMPLEX *b,
				SCALAR dt, read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				SCALAR kvector = get_float_from_image(kvectors, i, j, k % ${c.nvz});
				SCALAR prop_angle = kvector * dt / 2;
				COMPLEX prop_coeff = complex_ctr(cos(prop_angle), sin(prop_angle));

				COMPLEX a0 = a[index];
				COMPLEX b0 = b[index];

				a[index] = complex_mul(a0, prop_coeff);
				b[index] = complex_mul(b0, prop_coeff);
			}

			// Propagates state vector in x-space for evolution calculation
			%for suffix in ('', 'Wigner'):
			EXPORTED_FUNC void propagateXSpaceTwoComponent${suffix}(GLOBAL_MEM COMPLEX *aa,
				GLOBAL_MEM COMPLEX *bb, SCALAR dt,
				read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				SCALAR V = get_float_from_image(potentials, i, j, k % ${c.nvz});

				COMPLEX a = aa[index];
				COMPLEX b = bb[index];

				//store initial x-space field
				COMPLEX a0 = a;
				COMPLEX b0 = b;

				COMPLEX pa, pb, da = complex_ctr(0, 0), db = complex_ctr(0, 0);
				SCALAR n_a, n_b;

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					n_a = squared_abs(a);
					n_b = squared_abs(b);

					// TODO: there must be no minus sign before imaginary part,
					// but without it the whole thing diverges
					pa = complex_ctr(
						-(${c.l111} * n_a * n_a + ${c.l12} * n_b) / 2,
						-(-V - ${c.g11} * n_a - ${c.g12} * n_b));
					pb = complex_ctr(
						-(${c.l22} * n_b + ${c.l12} * n_a) / 2,
						-(-V - ${c.g22} * n_b - ${c.g12} * n_a));

					%if suffix == "Wigner":
						pa += complex_ctr(
							(1.5 * n_a - 0.75 / ${c.dV}) * ${c.l111} + ${c.l12} * 0.25,
							-(${c.g11} + 0.5 * ${c.g12})) / ${c.dV};
						pb += complex_ctr(
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

		self._program = self._env.compileProgram(kernels, self._constants)
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
			if self._constants.dim == 3:
				data1[start:stop,:,:] *= kcoeff
				data2[start:stop,:,:] *= kcoeff
			else:
				data1[start:stop] *= kcoeff
				data2[start:stop] *= kcoeff

	def _gpu__xpropagate(self, cloud, dt):
		if cloud.type == WIGNER:
			func = self._xpropagate_wigner
		else:
			func = self._xpropagate_func

		func(cloud.a.shape, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(dt), self._potentials)

	def _cpu__xpropagate(self, cloud, dt):
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

		p = self._potentials * 1j
		nvz = self._constants.nvz

		for iter in xrange(self._constants.itmax):
			n_a = numpy.abs(a.data) ** 2
			n_b = numpy.abs(b.data) ** 2

			pa = n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) + \
				1j * (n_a * g11_by_hbar + n_b * g12_by_hbar)

			pb = n_b * (-l22 / 2) + n_a * (-l12 / 2) + \
				1j * (n_b * g22_by_hbar + n_a * g12_by_hbar)

			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				pa[start:stop] += p
				pb[start:stop] += p

			da = numpy.exp(pa * (dt / 2))
			db = numpy.exp(pb * (dt / 2))

			a.data = a0 * da
			b.data = b0 * db

		a.data *= da
		b.data *= db

	def _noiseFunc(self, a_data, b_data, dt):
		coeff = math.sqrt(dt / self._constants.dV)

		n1 = numpy.abs(a_data) ** 2
		n2 = numpy.abs(b_data) ** 2
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = 0.125 * l12 * n2 + 0.375 * l111 * n1 * n1
		d = 0.125 * l12 * n1 + 0.25 * l22 * n2
		t = 0.25 * l12 * a_data * numpy.conj(b_data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		k1 = numpy.nan_to_num(c / t2)
		k2 = numpy.nan_to_num(t3)
		k3 = numpy.nan_to_num(b / t1)
		k4 = numpy.nan_to_num(t2)
		k5 = numpy.nan_to_num(-c / t1)
		k6 = numpy.nan_to_num(b * c / (a * t2))
		k7 = numpy.nan_to_num(b / a * t3)
		k8 = numpy.nan_to_num(numpy.sqrt((a ** 2 - b ** 2 - c ** 2) / a))

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)

		return \
			coeff * t1 + 1j * zeros, \
			zeros + 1j * coeff * k1, \
			zeros + 1j * coeff * k2, \
			zeros + 1j * zeros, \
			coeff * (k3 + 1j * k5), \
			coeff * (k4 + 1j * k6), \
			coeff * (zeros + 1j * k7), \
			coeff * (zeros + 1j * k8)

	def _propagateNoise(self, cloud, dt):

		shape = self._constants.ens_shape
		Z1 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]
		Z0 = [numpy.random.normal(scale=1, size=shape).astype(
			self._constants.scalar.dtype) for i in xrange(4)]

		G101, G102, G103, G104, G201, G202, G203, G204 = self._noiseFunc(cloud.a.data, cloud.b.data, dt)
		G111, G112, G113, G114, G211, G212, G213, G214 = \
			self._noiseFunc(
				cloud.a.data + math.sqrt(dt / 2) * (
					G101 * Z1[0] + G102 * Z1[1] + G103 * Z1[2] + G104 * Z1[3]),
				cloud.b.data + math.sqrt(dt / 2) * (
					G201 * Z1[0] + G202 * Z1[1] + G203 * Z1[2] + G204 * Z1[3]), dt)

		G121, G122, G123, G124, G221, G222, G223, G224 = \
			self._noiseFunc(
				cloud.a.data + math.sqrt(dt / 2) * (
					G101 * Z1[0] + G102 * Z1[1] + G103 * Z1[2] + G104 * Z1[3]),
				cloud.b.data + math.sqrt(dt / 2) * (
					G201 * Z1[0] + G202 * Z1[1] + G203 * Z1[2] + G204 * Z1[3]), dt)

		cloud.a.data += 0.5 * math.sqrt(dt) * (
				(G111 + G121) * Z0[0] +
				(G112 + G122) * Z0[1] +
				(G113 + G123) * Z0[2] +
				(G114 + G124) * Z0[3]
			)

		cloud.b.data += 0.5 * math.sqrt(dt) * (
				(G211 + G221) * Z0[0] +
				(G212 + G222) * Z0[1] +
				(G213 + G223) * Z0[2] +
				(G214 + G224) * Z0[3]
			)

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
		if self._noise:
			self._propagateNoise(cloud, dt)
		cloud.time += dt

		self._midstep = True
		self._toKSpace(cloud)
		self._projector(cloud.a.data)
		self._projector(cloud.b.data)

	def _runCallbacks(self, t, cloud, callbacks):
		if callbacks is None:
			return

		self._finishStep(cloud, self._constants.dt_evo)
		self._toXSpace(cloud)
		for callback in callbacks:
			callback(t, cloud)
		self._toKSpace(cloud)

	def run(self, cloud, time, callbacks=None, callback_dt=0, noise=True):

		# in SI units
		t = 0
		callback_t = 0
		self._noise = noise

		# in natural units
		dt = self._constants.dt_evo

		self._toKSpace(cloud)

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:
				self.propagate(cloud, dt)
				t += dt
				callback_t += dt

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

			self._toXSpace(cloud)

		except TerminateEvolution:
			return t
