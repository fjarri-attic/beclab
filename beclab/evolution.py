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

			INTERNAL_FUNC void noiseFunc(COMPLEX *G, COMPLEX a0, COMPLEX b0, SCALAR dt)
			{
				<%
					coeff = sqrt(1.0 / c.dV)
					k111 = c.l111 / 6.0
					k12 = c.l12 / 2.0
					k22 = c.l22 / 4.0
				%>

				COMPLEX gamma111_1 = complex_mul_scalar(complex_mul(a0, a0),
					(SCALAR)${sqrt(k111 / 2.0) * 3.0 * coeff});
				COMPLEX gamma12_1 = complex_mul_scalar(b0,
					(SCALAR)${sqrt(k12 / 2.0) * coeff});
				COMPLEX gamma12_2 = complex_mul_scalar(a0,
					(SCALAR)${sqrt(k12 / 2.0) * coeff});
				COMPLEX gamma22_2 = complex_mul_scalar(b0,
					(SCALAR)${sqrt(k22 / 2.0) * 2.0 * coeff});

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

				%for i in xrange(3):
				COMPLEX Z${i} = randoms[index + ${c.cells * c.ensembles * i}];
				%endfor

				SCALAR sdt2 = sqrt(dt / (SCALAR)2.0);
				SCALAR sdt = sqrt(dt);

				noiseFunc(G, a0, b0, dt);

				noiseFunc(G,
					a0 + complex_mul_scalar(
						complex_mul_scalar(G[0], Z0.y) +
						complex_mul_scalar(G[1], Z1.y) +
						complex_mul_scalar(G[2], Z2.y), sdt2),
					b0 + complex_mul_scalar(
						complex_mul_scalar(G[3], Z0.y) +
						complex_mul_scalar(G[4], Z1.y) +
						complex_mul_scalar(G[5], Z2.y), sdt2), dt);

				a[index] = a0 + complex_mul_scalar(
						complex_mul_scalar(G[0], Z0.x) +
						complex_mul_scalar(G[1], Z1.x) +
						complex_mul_scalar(G[2], Z2.x), sdt);

				b[index] = b0 + complex_mul_scalar(
						complex_mul_scalar(G[3], Z0.x) +
						complex_mul_scalar(G[4], Z1.x) +
						complex_mul_scalar(G[5], Z2.x), sdt);
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

	def _noiseFunc(self, a_data, b_data):
		coeff = math.sqrt(1.0 / self._constants.dV)
		k111 = self._constants.l111 / 6.0
		k12 = self._constants.l12 / 2.0
		k22 = self._constants.l22 / 4.0

		gamma111_1 = a_data ** 2 * (math.sqrt(k111 / 2.0) * 3.0 * coeff)
		gamma12_1 = b_data * (math.sqrt(k12 / 2.0) * coeff)
		gamma12_2 = a_data * (math.sqrt(k12 / 2.0) * coeff)
		gamma22_2 = b_data * (math.sqrt(k22 / 2.0) * 2.0 * coeff)

		zeros = numpy.zeros(a_data.shape, self._constants.scalar.dtype)

		return gamma111_1, gamma12_1, zeros, zeros, gamma12_2, gamma22_2

	def _cpu__propagateNoise(self, cloud, dt):
		shape = self._constants.ens_shape
		r_shape = tuple([3] + list(shape))

		randoms = self._random.random_normal(size=cloud.a.size * 3).reshape(*r_shape)

		G00, G01, G02, G03, G04, G05 = self._noiseFunc(cloud.a.data, cloud.b.data)
		G10, G11, G12, G13, G14, G15 = self._noiseFunc(
			cloud.a.data + math.sqrt(dt / 2.0) * (
				G00 * randoms[0].real + G01 * randoms[1].real + G02 * randoms[2].real),
			cloud.b.data + math.sqrt(dt / 2.0) * (
				G03 * randoms[0].real + G04 * randoms[1].real + G05 * randoms[2].real)
		)

		cloud.a.data += math.sqrt(dt) * (G10 * randoms[0].imag +
			G11 * randoms[1].imag + G12 * randoms[2].imag)
		cloud.b.data += math.sqrt(dt) * (G13 * randoms[0].imag +
			G14 * randoms[1].imag + G15 * randoms[2].imag)

	def _gpu__propagateNoise(self, cloud, dt):
		randoms = self._random.random_normal(size=cloud.a.size * 3)

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
			self._propagateNoise(cloud, dt)

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

		starting_time = cloud.time
		callback_t = 0

		# in natural units
		dt = self._constants.dt_evo

		self._toKSpace(cloud)

		try:
			self._runCallbacks(cloud.time, cloud, callbacks)

			while cloud.time - starting_time < time:
				self.propagate(cloud, dt, noise)

				callback_t += dt

				if callback_t > callback_dt:
					self._runCallbacks(cloud.time, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(cloud.time, cloud, callbacks)

			self._toXSpace(cloud)

		except TerminateEvolution:
			return cloud.time
