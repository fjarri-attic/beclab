import numpy
import math

from .helpers import *
from .constants import REPR_CLASSICAL, REPR_WIGNER
from .meters import ProjectionMeter, UncertaintyMeter, getXiSquared
from .evolution import TerminateEvolution
from .pulse import Pulse
from .wavefunction import WavefunctionSet


class ParticleNumberCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False, pulse=None):
		PairedCalculation.__init__(self, env)
		self.stats = ParticleStatistics(env, constants, grid)
		self.verbose = verbose
		self._pulse = pulse

		self.times = []
		self.N = []

		self._psi = WavefunctionSet(env, constants, grid)
		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self.stats.prepare(components=self._p.components, ensembles=self._p.ensembles,
			psi_type=self._p.psi_type)
		self._psi.prepare(components=self._p.components, ensembles=self._p.ensembles)

	def __call__(self, t, dt, psi):
		psi.copyTo(self._psi)

		if self._pulse is not None:
			self._pulse.apply(self._psi, theta=0.5 * numpy.pi)

		N = self.stats.getN(self._psi)
		if self.verbose:
			print "Particle counter: ", t, "s,", N, N.sum()

		self.times.append(t)
		self.N.append(N)

	def getData(self):
		N = numpy.array(self.N)
		return numpy.array(self.times), N.transpose(), N.sum(1)


class ParticleNumberCondition(ParticleNumberCollector):

	def __init__(self, env, constants, grid, ratio=0.5, **kwds):
		ParticleNumberCollector.__init__(self, env, constants, grid, **kwds)
		self._previous_ratio = None
		self._ratio = ratio

	def __call__(self, t, dt, psi):
		ParticleNumberCollector.__call__(self, t, dt, psi)

		ratio = self.N[-1][0] / (self.N[-1][0] + self.N[-1][1])

		if self._previous_ratio is None:
			self._previous_ratio = ratio

		if (ratio > self._ratio and self._previous_ratio < self._ratio) or \
				(ratio < self._ratio and self._previous_ratio > self._ratio):
			raise TerminateEvolution()


class PhaseNoiseCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False):
		PairedCalculation.__init__(self, env)
		self._stats = ParticleStatistics(env, constants, grid)
		self.times = []
		self.phnoise = []
		self._verbose = verbose

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._stats.prepare(components=self._p.components, ensembles=self._p.ensembles,
			psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		phnoise = self._stats.getPhaseNoise(psi)

		if self._verbose:
			print "Phase noise:", t, "s,", phnoise

		self.times.append(t)
		self.phnoise.append(phnoise)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.phnoise)


class PzNoiseCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False):
		PairedCalculation.__init__(self, env)
		self._stats = ParticleStatistics(env, constants, grid)
		self.times = []
		self.pznoise = []
		self._verbose = verbose

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._stats.prepare(components=self._p.components, ensembles=self._p.ensembles,
			psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		pznoise = self._stats.getPzNoise(psi)

		if self._verbose:
			print "Pz noise:", t, "s,", pznoise

		self.times.append(t)
		self.pznoise.append(pznoise)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.pznoise)


class VisibilityCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False):
		PairedCalculation.__init__(self, env)
		self.stats = ParticleStatistics(env, constants, grid)
		self.verbose = verbose

		self.times = []
		self.visibility = []

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self.stats.prepare(components=self._p.components,
			ensembles=self._p.ensembles, psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		v = self.stats.getVisibility(psi)

		if self.verbose:
			print "Visibility: ", t, "s,", v

		self.times.append(t)
		self.visibility.append(v)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.visibility)


class SurfaceProjectionCollector(PairedCalculation):

	def __init__(self, env, constants, grid, pulse=None):
		PairedCalculation.__init__(self, env)
		self._projection = DensityProfile(env, constants, grid)
		self._pulse = pulse
		self._constants = constants
		self._grid = grid

		self._psi = WavefunctionSet(env, constants, grid)

		self.times = []
		self.xy = []
		self.yz = []

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._projection.prepare(components=self._p.components,
			ensembles=self._p.ensembles, psi_type=self._p.psi_type)
		self._psi.prepare(components=self._p.components, ensembles=self._p.ensembles)

	def __call__(self, t, dt, psi):
		psi.copyTo(self._psi)

		if self._pulse is not None:
			self._pulse.apply(self._psi, theta=0.5 * numpy.pi)

		self.times.append(t)
		self.xy.append(self._projection.getXY(self._psi))
		self.yz.append(self._projection.getYZ(self._psi))

	def getData(self):
		shape = (len(self.times), self._p.components)
		shape_xy = shape + (self._grid.shape[1:3])
		shape_yz = shape + (self._grid.shape[:2])
		return numpy.array(self.times), \
			numpy.transpose(numpy.concatenate(self.xy).reshape(*shape_xy), axes=(1, 0, 2, 3)), \
			numpy.transpose(numpy.concatenate(self.yz).reshape(*shape_yz), axes=(1, 0, 2, 3))


class AxialProjectionCollector(PairedCalculation):

	def __init__(self, env, constants, grid, pulse=None):
		PairedCalculation.__init__(self, env)
		self._projection = DensityProfile(env, constants, grid)
		self._pulse = pulse
		self._constants = constants
		self._grid = grid

		self.times = []
		self.snapshots = []
		self._psi = WavefunctionSet(env, constants, grid)

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._projection.prepare(components=self._p.components,
			ensembles=self._p.ensembles, psi_type=self._p.psi_type)
		self._psi.prepare(components=self._p.components, ensembles=self._p.ensembles)

	def __call__(self, t, dt, psi):

		psi.copyTo(self._psi)

		if self._pulse is not None:
			self._pulse.apply(self._psi, theta=0.5 * numpy.pi)

		self.times.append(t)

		proj = self._projection.getZ(self._psi)

		self.snapshots.append((proj[0] - proj[1]) / (proj[0] + proj[1]))

	def getData(self):
		return numpy.array(self.times), \
			numpy.concatenate(self.snapshots).reshape(
				len(self.times), self.snapshots[0].size).transpose()


class UncertaintyCollector(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._unc = Uncertainty(env, constants, grid)
		self.times = []
		self.N_stddev = []
		self.XiSquared = []

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._unc.prepare(components=self._p.components,
			ensembles=self._p.ensembles, psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		self.times.append(t)
		self.N_stddev.append(self._unc.getNstddev(psi))
		i, n = self._unc.getEnsembleSums(psi)
		self.XiSquared.append(getXiSquared(i, n[0], n[1]))

	def getData(self):
		return numpy.array(self.times), \
			numpy.array(self.N_stddev).transpose(), \
			numpy.array(self.XiSquared)


class SpinCloudCollector:

	def __init__(self, env, constants, grid):
		self._unc = Uncertainty(env, constants, grid)
		self.times = []
		self.phi = []
		self.yps = []

	def prepare(self, **kwds):
		self._unc.prepare(components=kwds['components'],
			ensembles=kwds['ensembles'], psi_type=kwds['psi_type'])

	def __call__(self, t, dt, psi):
		self.times.append(t)
		i, n = self._unc.getEnsembleSums(psi)
		phi, yps = getSpins(i, n[0], n[1])
		self.phi.append(phi)
		self.yps.append(yps)

	def getData(self):
		return [numpy.array(x) for x in (self.times, self.phi, self.yps)]


class AnalyticNoiseCollector:
	"""
	According to Ueda and Kitagawa, http://link.aps.org/doi/10.1103/PhysRevA.47.5138, (5)
	"""

	def __init__(self, env, constants, grid):
		self._stats = ParticleStatistics(env, constants, grid)
		self._env = env
		self._constants = constants
		self._grid = grid
		self.times = []
		self.noise = []

	def prepare(self, **kwds):
		self._stats.prepare(components=kwds['components'],
			ensembles=kwds['ensembles'], psi_type=kwds['psi_type'])

	def __call__(self, t, dt, psi):
		n = self._env.fromDevice(self._stats.getAverageDensity(psi))
		dV = self._grid.dV

		comp1 = 0
		comp2 = 1
		g = self._constants.g

		n1 = n[comp1]
		n2 = n[comp2]

		N1 = (n1 * dV).sum()
		N2 = (n2 * dV).sum()
		N = N1 + N2

		n1 /= N1
		n2 /= N2

		chi = 1.0 / (2.0 * self._constants.hbar) * (
			g[comp1, comp1] * (n1 * n1 * dV).sum() +
			g[comp2, comp2] * (n2 * n2 * dV).sum() -
			2 * g[comp1, comp2] * (n1 * n2 * dV).sum())

		self.times.append(t)
		self.noise.append(numpy.sqrt(N) * chi * t)

	def getData(self):
		return [numpy.array(x) for x in (self.times, self.noise)]
