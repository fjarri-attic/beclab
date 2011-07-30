import numpy
from .helpers import *
from .evolution import SplitStepEvolution


class Pulse(PairedCalculation):

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self._addParameters(components=2, f_detuning=0, starting_phase=0, ensembles=1,
			f_rabi=0)
		self.prepare(**kwds)

	def _prepare(self):
		assert self._p.components >= 2 # TODO: currently will just apply pulse to the first two components
		self._p.w_detuning = self._p.f_detuning * 2.0 * numpy.pi
		self._p.w_rabi = self._p.f_rabi * 2.0 * numpy.pi
		self._p.comp_size = self._grid.size * self._p.ensembles

	def _gpu__prepare_specific(self):
		kernels = """
			<%!
				from math import sqrt
			%>

			EXPORTED_FUNC void matrix_pulse(int gsize, GLOBAL_MEM COMPLEX *data,
				SCALAR theta, SCALAR phi)
			{
				COMPLEX a0 = data[GLOBAL_INDEX];
				COMPLEX b0 = data[GLOBAL_INDEX + ${p.comp_size}];

				SCALAR sin_half_theta = sin(theta / (SCALAR)2.0);
				SCALAR cos_half_theta = cos(theta / (SCALAR)2.0);

				COMPLEX minus_i = complex_ctr(0, -1);

				COMPLEX k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, -phi))
				), sin_half_theta);

				COMPLEX k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, phi))
				), sin_half_theta);

				data[GLOBAL_INDEX] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				data[GLOBAL_INDEX + ${p.comp_size}] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}

			EXPORTED_FUNC void matrix_pulse_noise(int gsize, GLOBAL_MEM COMPLEX *data,
				SCALAR *thetas, SCALAR *phis)
			{
				COMPLEX a0 = data[GLOBAL_INDEX];
				COMPLEX b0 = data[GLOBAL_INDEX + ${p.comp_size}];

				int ensemble = GLOBAL_INDEX / ${g.size};
				SCALAR theta = thetas[ensemble];
				SCALAR phi = phis[ensemble];

				SCALAR sin_half_theta = sin(theta / (SCALAR)2.0);
				SCALAR cos_half_theta = cos(theta / (SCALAR)2.0);

				COMPLEX minus_i = complex_ctr(0, -1);

				COMPLEX k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, -phi))
				), sin_half_theta);

				COMPLEX k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(complex_ctr(0, phi))
				), sin_half_theta);

				data[GLOBAL_INDEX] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				data[GLOBAL_INDEX + ${p.comp_size}] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}
		"""

		self.__program = self.compileProgram(kernels)
		self._kernel_matrix_pulse = self.__program.matrix_pulse
		self._kernel_matrix_pulse_noise = self.__program.matrix_pulse_noise

	def _cpu__kernel_matrix_pulse(self, gsize, data, theta, phi):
		a0 = data[0].copy()
		b0 = data[1].copy()

		half_theta = theta / 2.0
		k1 = numpy.cos(half_theta)
		k2 = -1j * numpy.exp(-1j * phi) * numpy.sin(half_theta)
		k3 = -1j * numpy.exp(1j * phi) * numpy.sin(half_theta)

		data[0].flat[:] = (a0 * k1 + b0 * k2).flat
		data[1].flat[:] = (a0 * k3 + b0 * k1).flat

	def _cpu__kernel_matrix_pulse_noise(self, gsize, data, thetas, phis):

		ens = self._p.ensembles

		a0 = data[0].copy()
		b0 = data[1].copy()

		half_thetas = thetas / 2.0
		k1 = numpy.cos(half_thetas)
		k2 = -1j * numpy.exp(-1j * phis) * numpy.sin(half_thetas)
		k3 = -1j * numpy.exp(1j * phis) * numpy.sin(half_thetas)

		f = lambda x: x.repeat(self._grid.size).reshape((ens,) + self._grid.shape)
		k1 = f(k1)
		k2 = f(k2)
		k3 = f(k3)

		data[0] = a0 * k1 + b0 * k2
		data[1] = a0 * k3 + b0 * j1

	def _applyMatrix(self, psi, theta, phi):
		self._kernel_matrix_pulse(psi.size, psi.data,
			self._constants.scalar.cast(theta),
			self._constants.scalar.cast(phi))

	def _applyMatrixNoise(self, psi, theta, phi, theta_noise, phi_noise):

		ens = self._p.ensembles
		dtype = self._constants.scalar.dtype

		# TODO: use GPU random
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

		self._kernel_matrix_pulse_noise(psi.size, psi.data, d_thetas, d_phis)

	def _gpu__calculateNoiseMatrix(self, a_data, b_data, thetas, phis):
		return self._calculateNoiseMatrixFunc(a_data.size, a_data, b_data, thetas, phis)

	def apply(self, psi, theta, matrix=True, theta_noise=0.0, phi_noise=0.0):
		self.prepare(components=psi.components, ensembles=psi.ensembles)

		phi = psi.time * self._p.w_detuning + self._p.starting_phase

		t_pulse = (theta / numpy.pi / 2.0) / self._p.f_rabi

		if phi_noise > 0 or theta_noise > 0:
			self._applyMatrixNoise(psi, theta, phi, theta_noise, phi_noise)
		elif matrix:
			self._applyMatrix(psi, theta, phi)
		else:
			raise NotImplementedError()
			#self._evolution.run(cloud, t_pulse, starting_phase=phi)

		psi.time += t_pulse
