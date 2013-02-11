"""
Classes, modeling the evolution of BEC.
"""

import numpy
import copy

from .helpers import *
from .constants import *
from .ground_state import RK5Propagation, Projector
from .wavefunction import WavefunctionSet


class TerminateEvolution(Exception):
	pass


class StrongEvolution(PairedCalculation):

	def __init__(self, env):
		PairedCalculation.__init__(self, env)

	def _toMeasurementSpace(self, psi):
		pass

	def _toEvolutionSpace(self, psi):
		pass

	def _collectMetrics(self, t):
		pass

	def _runCallbacks(self, psi, callbacks, dt):
		if callbacks is None:
			return

		self._toMeasurementSpace(psi)
		for callback in callbacks:
			callback(psi.time, dt, psi)
		self._toEvolutionSpace(psi)

	def _run_step(self, psi, interval, steps,
			callbacks=None, samples=1, starting_phase=0, double_step=False):

		dt = self._constants.scalar.cast(interval) / (steps / (2 if double_step else 1))
		noise_dt = dt / (2 if double_step else 1)

		if not double_step:
			self._runCallbacks(psi, callbacks, dt)

		starting_time = psi.time

		self._toEvolutionSpace(psi)

		for step in xrange(steps / (2 if double_step else 1)):

			#try:
			psi.time = starting_time + step * dt
			self.propagate(psi, psi.time, dt, noise_dt, double_step=double_step)

			#except TerminateEvolution:
			#	final_time = psi.time

			if (step + 1) % ((steps / (2 if double_step else 1)) / samples) == 0:
				if double_step:
					print "Skipping callbacks at t =", psi.time
				else:
					self._runCallbacks(psi, callbacks, dt)

		self._toMeasurementSpace(psi)


	def run(self, psi, interval, steps, callbacks=None, samples=1, starting_phase=0):

		assert steps % samples == 0
		assert steps % 2 == 0

		if callbacks is not None:
			for cb in callbacks:
				cb.prepare(components=psi.components, ensembles=psi.ensembles,
					psi_type=psi.type)

		self._phi = starting_phase
		self.prepare(ensembles=psi.ensembles, components=psi.components,
			psi_type=psi.type)

		starting_time = psi.time

		# copy the initial state before propagation
		psi_double = psi.copy()

		# propagate with normal step
		print "--- Normal step"
		if hasattr(self, '_random'):
			rng_state = self._random.get_state()
		self._run_step(psi, interval, steps,
			callbacks=callbacks, samples=samples,
			starting_phase=starting_phase, double_step=False)

		# propagate with the double step
		print "--- Double step"
		if hasattr(self, '_random'):
			self._random.set_state(rng_state)
		self._run_step(psi_double, interval, steps,
			samples=samples,
			starting_phase=starting_phase, double_step=True)

		pd = self._env.fromDevice(psi_double.data)
		p = self._env.fromDevice(psi.data)

		p = p.transpose(1, 0, *(range(2, len(p.shape))))
		pd = pd.transpose(1, 0, *(range(2, len(pd.shape))))

		diff = numpy.abs(p - pd)
		norm = numpy.abs(p)

		diff_norms = numpy.array([numpy.linalg.norm(diff[i]) for i in xrange(self._p.ensembles)])
		norm_norms = numpy.array([numpy.linalg.norm(norm[i]) for i in xrange(self._p.ensembles)])

		print "Propagation error (max over ensembles, mean over space)", \
			(diff_norms / norm_norms).max()



class StrongRKEvolution(StrongEvolution):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants, grid, **kwds):
		assert isinstance(grid, UniformGrid)

		StrongEvolution.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self._kvectors = grid.energy_device

		self._projector = Projector(env, constants, grid)
		self._random = createRandom(env, constants.double)

		self._addParameters(f_rabi=0, f_detuning=0, noise=False,
			ensembles=1, components=2, psi_type=REPR_CLASSICAL, potentials=None)
		self.prepare(**kwds)

		self.ai = numpy.array([0.0, -0.737101392796, -1.634740794341,
			-0.744739003780, -1.469897351522, -2.813971388035])
		self.bi = numpy.array([0.032918605146, 0.823256998200, 0.381530948900,
			0.200092213184, 1.718581042715, 0.27])
		self.ci = numpy.array([0.0, 0.032918605146, 0.249351723343,
			0.466911705055, 0.582030414044, 0.847252983783])


	def _prepare(self):
		# FIXME: matrix exponent in xpropagate() requires 2 components
		# different number will require significant changes
		assert self._p.components == 2

		if self._p.potentials is None:
			self._p.separate_potentials = False
			self._potentials = self._grid.potentials_device
		else:
			self._p.separate_potentials = True
			self._potentials = self._env.toDevice(self._p.potentials)

		self._projector.prepare(components=self._p.components, ensembles=self._p.ensembles)

		self._p.w_detuning = 2 * numpy.pi * self._p.f_detuning
		self._p.w_rabi = 2 * numpy.pi * self._p.f_rabi

		self._p.grid_size = self._grid.size
		self._p.comp_size = self._grid.size * self._p.ensembles

		self._p.g = self._constants.g / self._constants.hbar

		self._p.losses_drift = copy.deepcopy(self._constants.losses_drift)

		# FIXME: add support for non-uniform grids
		dV = self._grid.dV_uniform
		grid_shape = self._grid.shape
		self._normalization = numpy.sqrt(1.0 / dV)
		self._p.losses_diffusion = copy.deepcopy(self._constants.losses_diffusion)
		self._randoms1 = self._env.allocate(
			(self._constants.noise_sources, self._p.ensembles) + grid_shape,
			dtype=self._constants.complex.dtype
		)
		self._randoms2 = self._env.allocate(
			(self._constants.noise_sources, self._p.ensembles) + grid_shape,
			dtype=self._constants.complex.dtype
		)

		self._psi_kspace = WavefunctionSet(self._env, self._constants, self._grid)
		self._psi_omega = WavefunctionSet(self._env, self._constants, self._grid)
		self._psi_omega.fillWithValue(0)
		self._psi_kspace.createEnsembles(self._p.ensembles)
		self._psi_omega.createEnsembles(self._p.ensembles)


	def _gpu__prepare_specific(self):

		kernels = """
			<%!
				from math import sqrt, pi
			%>

			// Propagates state vector in k-space
			EXPORTED_FUNC void kpropagate(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *kvectors)
			{
				LIMITED_BY(${p.comp_size});

				SCALAR k = kvectors[GLOBAL_INDEX % ${p.grid_size}];
				COMPLEX mode_coeff = complex_ctr(0, -k);
				COMPLEX val;

				%for comp in xrange(p.components):
					val = data[GLOBAL_INDEX + ${comp * p.comp_size}];
					data[GLOBAL_INDEX + ${comp * p.comp_size}] = complex_mul(
						val, mode_coeff
					);
				%endfor
			}

			// Propagates state vector in x-space
			EXPORTED_FUNC void xpropagate(int gsize,
				GLOBAL_MEM COMPLEX *omega,
				GLOBAL_MEM COMPLEX *psi,
				GLOBAL_MEM COMPLEX *psik,
				GLOBAL_MEM SCALAR *potentials,
				GLOBAL_MEM COMPLEX *noise1,
				GLOBAL_MEM COMPLEX *noise2,
				SCALAR ai,
				SCALAR t, SCALAR dt, SCALAR phi, int double_step)
			{
				LIMITED_BY(${p.comp_size});

				%if p.separate_potentials:
				%for comp in xrange(p.components):
				SCALAR V${comp} = potentials[GLOBAL_INDEX % ${p.grid_size} + ${p.grid_size * comp}];
				%endfor
				%else:
				SCALAR V = potentials[GLOBAL_INDEX % ${p.grid_size}];
				%for comp in xrange(p.components):
				SCALAR V${comp} = V;
				%endfor
				%endif

				%if p.psi_type == REPR_WIGNER:
				COMPLEX noises[${c.noise_sources}];
				if (double_step)
				{
					%for s in xrange(c.noise_sources):
					noises[${s}] = complex_mul_scalar(noise1[GLOBAL_INDEX + ${s * p.comp_size}] +
						noise2[GLOBAL_INDEX + ${s * p.comp_size}], 0.5);
					%endfor
				}
				else
				{
					%for s in xrange(c.noise_sources):
					noises[${s}] = noise1[GLOBAL_INDEX + ${s * p.comp_size}];
					%endfor
				}
				%endif

				%for comp in xrange(p.components):
				COMPLEX val${comp} = psi[GLOBAL_INDEX + ${comp * p.comp_size}];
				COMPLEX valk${comp} = psik[GLOBAL_INDEX + ${comp * p.comp_size}];
				SCALAR n${comp} = squared_abs(val${comp});
				%endfor

				<%
					def complex_mul_sequence(s):
						if len(s) == 1:
							return s[0]
						else:
							return 'complex_mul(' + s[0] + ', ' + complex_mul_sequence(s[1:]) + ')'
				%>

				%for comp in xrange(p.components):

				COMPLEX N${comp} =
					valk${comp} +
					complex_mul(
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
					-V${comp}
					%for comp_other in xrange(p.components):
					-(SCALAR)${p.g[comp, comp_other]} * n${comp_other}
					%endfor
					), val${comp});

				%if p.psi_type == REPR_WIGNER:
				%for i, e in enumerate(p.losses_diffusion[comp]):
				<%
					coeff, orders = e
					sequence = []
					for j, order in enumerate(orders):
						if order > 0:
							sequence += ['val' + str(j)] * order
					if len(sequence) > 0:
						product = complex_mul_sequence(sequence)
					else:
						product = 'complex_ctr(1, 0)'
				%>
				%if coeff != 0:
				N${comp} = N${comp} + complex_mul(conj(complex_mul_scalar(
					${product},
					(SCALAR)${coeff}
				)), noises[${i}]);
				%endif
				%endfor
				%endif

				%endfor

				COMPLEX om;
				%for comp in xrange(p.components):
					om = omega[GLOBAL_INDEX + ${comp * p.comp_size}];
					omega[GLOBAL_INDEX + ${comp * p.comp_size}] =
						complex_mul_scalar(om, ai)
						+ complex_mul_scalar(N${comp}, dt);
				%endfor
			}


			EXPORTED_FUNC void resupdate(int gsize,
				GLOBAL_MEM COMPLEX *psi,
				GLOBAL_MEM COMPLEX *omega,
				SCALAR bi)
			{
				LIMITED_BY(${p.comp_size});

				%for comp in xrange(p.components):
				COMPLEX val${comp} = psi[GLOBAL_INDEX + ${comp * p.comp_size}];
				COMPLEX om${comp} = omega[GLOBAL_INDEX + ${comp * p.comp_size}];
				psi[GLOBAL_INDEX + ${comp * p.comp_size}] = val${comp}
					+ complex_mul_scalar(om${comp}, bi);
				%endfor
			}

		"""

		self.__program = self.compileProgram(kernels)
		self._kernel_kpropagate = self.__program.kpropagate
		self._kernel_xpropagate = self.__program.xpropagate
		self._kernel_resupdate = self.__program.resupdate

	def _toMeasurementSpace(self, psi):
		pass

	def _toEvolutionSpace(self, psi):
		pass

	def propagate(self, psi, t, dt, noise_dt, double_step=False):

		cast = self._constants.scalar.cast

		noise_scale = numpy.sqrt(1./noise_dt) * self._normalization

		self._random.random_normal(self._randoms1, scale=noise_scale)
		if double_step:
			self._random.random_normal(self._randoms2, scale=noise_scale)

		r1 = self._randoms1
		r2 = self._randoms2

		psik = self._psi_kspace
		psio = self._psi_omega

		for s in xrange(6):
			psi.copyTo(psik)
			psik.toMSpace()
			self._kernel_kpropagate(psik.size, psik.data, self._kvectors)
			psik.toXSpace()

			self._kernel_xpropagate(psi.size, psio.data, psi.data, psik.data,
				self._potentials, r1, r2,
				cast(self.ai[s]),
				cast(t), cast(dt), cast(self._phi), numpy.int32(double_step))

			psio.toMSpace()
			self._projector(psio.data)
			psio.toXSpace()

			self._kernel_resupdate(psi.size, psi.data, psio.data, cast(self.bi[s]))


