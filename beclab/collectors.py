import numpy
import math

from helpers import *
from .state import ParticleStatistics, Projection, Slice, Uncertainty
from .evolution import TerminateEvolution
from .pulse import Pulse


class ParticleNumberCollector:

	def __init__(self, env, constants, verbose=False, pulse=None, matrix_pulse=True):
		self.stats = ParticleStatistics(env, constants)
		self.verbose = verbose
		self._pulse = pulse
		self._matrix_pulse = matrix_pulse

		self.times = []
		self.Na = []
		self.Nb = []

	def __call__(self, t, cloud):
		cloud = cloud.copy(prepare=False)

		if self._pulse is not None:
			self._pulse.apply(cloud, theta=0.5 * math.pi, matrix=self._matrix_pulse)

		Na = self.stats.countParticles(cloud.a)
		Nb = self.stats.countParticles(cloud.b)
		if self.verbose:
			print "Particle counter: " + str((t, int(Na), int(Nb), int(Na + Nb)))

		self.times.append(t)
		self.Na.append(Na)
		self.Nb.append(Nb)

	def getData(self):
		Na = numpy.array(self.Na)
		Nb = numpy.array(self.Nb)
		return numpy.array(self.times), Na, Nb, Na + Nb


class ParticleNumberCondition:

	def __init__(self, env, constants, verbose=False, pulse=None, matrix_pulse=True, ratio=0.5):
		self._stats = ParticleStatistics(env, constants)
		self._verbose = verbose
		self._pulse = pulse
		self._matrix_pulse = matrix_pulse
		self._ratio = ratio

		self._previous_Na = None
		self._previous_ratio = None

	def __call__(self, t, cloud):
		cloud = cloud.copy(prepare=False)

		if self._pulse is not None:
			self._pulse.apply(cloud, theta=0.5 * math.pi, matrix=self._matrix_pulse)

		Na = self._stats.countParticles(cloud.a)
		Nb = self._stats.countParticles(cloud.b)

		ratio = Na / (Na + Nb)

		if self._verbose:
			print "Particle ratio: " + str((t, Na, Nb, ratio))

		if self._previous_ratio is None:
			self._previous_ratio = ratio

		if (ratio > self._ratio and self._previous_ratio < self._ratio) or \
				(ratio < self._ratio and self._previous_ratio > self._ratio):
			raise TerminateEvolution()


class PhaseNoiseCollector:

	def __init__(self, env, constants, verbose=False):
		self._stats = ParticleStatistics(env, constants)
		self._constants = constants
		self._times = []
		self._var = []
		self._verbose = verbose

	def __call__(self, t, cloud):
		noise = self._stats.getPhaseNoise(cloud.a, cloud.b)

		if self._verbose:
			print "Phase noise: " + repr((t, noise))

		self._times.append(t)
		self._var.append(noise)

	def getData(self):
		return numpy.array(self._times), numpy.array(self._var)


class PzNoiseCollector:

	def __init__(self, env, constants, verbose=False):
		self._stats = ParticleStatistics(env, constants)
		self._constants = constants
		self._times = []
		self._var = []
		self._verbose = verbose

	def __call__(self, t, cloud):
		noise = self._stats.getPzNoise(cloud.a, cloud.b)

		if self._verbose:
			print "Pz noise: " + repr((t, noise))

		self._times.append(t)
		self._var.append(noise)

	def getData(self):
		return numpy.array(self._times), numpy.array(self._var)


class VisibilityCollector:

	def __init__(self, env, constants, verbose=False):
		self.stats = ParticleStatistics(env, constants)
		self.verbose = verbose

		self.times = []
		self.visibility = []

	def __call__(self, t, cloud):
		v = self.stats.getVisibility(cloud.a, cloud.b)

		if self.verbose:
			print "Visibility: " + str((t, v))

		self.times.append(t)
		self.visibility.append(v)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.visibility)


class SurfaceProjectionCollector:

	def __init__(self, env, constants, pulse=None, matrix_pulse=True):
		self._projection = Projection(env, constants)
		self._pulse = pulse
		self._matrix_pulse = matrix_pulse
		self._constants = constants

		self.times = []
		self.a_xy = []
		self.a_yz = []
		self.b_xy = []
		self.b_yz = []

	def __call__(self, t, cloud):
		"""Returns numbers in units (particles per square micrometer)"""

		cloud = cloud.copy()

		if self._pulse is not None:
			self._pulse.apply(cloud, theta=0.5 * math.pi, matrix=self._matrix_pulse)

		self.times.append(t)

		coeff_xy = self._constants.dz
		coeff_yz = self._constants.dx

		self.a_xy.append(self._projection.getXY(cloud.a) * coeff_xy)
		self.a_yz.append(self._projection.getYZ(cloud.a) * coeff_yz)
		self.b_xy.append(self._projection.getXY(cloud.b) * coeff_xy)
		self.b_yz.append(self._projection.getYZ(cloud.b) * coeff_yz)

	def getData(self):
		return self.times, self.a_xy, self.a_yz, self.b_xy, self.b_yz


class SliceCollector:

	def __init__(self, env, constants, pulse=None, matrix_pulse=True):
		self._slice = Slice(env, constants)
		self._pulse = pulse
		self._matrix_pulse = matrix_pulse
		self._constants = constants

		self.times = []
		self.a_xy = []
		self.a_yz = []
		self.b_xy = []
		self.b_yz = []

	def __call__(self, t, cloud):
		"""Returns numbers in units (particles per square micrometer)"""

		cloud = cloud.copy()

		if self._pulse is not None:
			self._pulse.apply(cloud, theta=0.5 * math.pi, matrix=self._matrix_pulse)

		self.times.append(t)

		coeff_xy = 1.0
		coeff_yz = 1.0

		self.a_xy.append(self._slice.getXY(cloud.a) * coeff_xy)
		self.a_yz.append(self._slice.getYZ(cloud.a) * coeff_yz)
		self.b_xy.append(self._slice.getXY(cloud.b) * coeff_xy)
		self.b_yz.append(self._slice.getYZ(cloud.b) * coeff_yz)

	def getData(self):
		return self.times, self.a_xy, self.a_yz, self.b_xy, self.b_yz


class AxialProjectionCollector:

	def __init__(self, env, constants, pulse=None, matrix_pulse=True):
		self._projection = Projection(env, constants)
		self._pulse = pulse
		self._matrix_pulse = matrix_pulse
		self._constants = constants

		self.times = []
		self.snapshots = []

	def __call__(self, t, cloud):

		cloud = cloud.copy()

		if self._pulse is not None:
			self._pulse.apply(cloud, theta=0.5 * math.pi, matrix=self._matrix_pulse)

		self.times.append(t)

		a_proj = self._projection.getZ(cloud.a)
		b_proj = self._projection.getZ(cloud.b)

		self.snapshots.append((a_proj - b_proj) / (a_proj + b_proj))

	def getData(self):
		return numpy.array(self.times), numpy.concatenate(self.snapshots).reshape(len(self.times), self.snapshots[0].size).transpose()


class UncertaintyCollector:

	def __init__(self, env, constants):
		self._unc = Uncertainty(env, constants)
		self.times = []
		self.Na_stddev = []
		self.Nb_stddev = []
		self.XiSquared = []

	def __call__(self, t, cloud):
		self.times.append(t)
		self.Na_stddev.append(self._unc.getNstddev(cloud.a))
		self.Nb_stddev.append(self._unc.getNstddev(cloud.b))
		self.XiSquared.append(self._unc.getXiSquared(cloud.a, cloud.b))

	def getData(self):
		return [numpy.array(x) for x in
			(self.times, self.Na_stddev, self.Nb_stddev, self.XiSquared)]


class SpinCloudCollector:

	def __init__(self, env, constants):
		self._unc = Uncertainty(env, constants)
		self.times = []
		self.clouds = []

	def __call__(self, t, cloud):
		self.times.append(t)
		self.clouds.append(self._unc.getSpins(cloud.a, cloud.b))

	def getData(self):
		return numpy.array(self.times), self.clouds

class AnalyticNoiseCollector:
	"""
	According to Ueda and Kitagawa, http://link.aps.org/doi/10.1103/PhysRevA.47.5138, (5)
	"""

	def __init__(self, env, constants):
		self._stats = ParticleStatistics(env, constants)
		self.times = []
		self.noise = []
		self._constants = constants
		self._env = env

	def __call__(self, t, cloud):
		n1 = self._env.fromDevice(self._stats.getAverageDensity(cloud.a))
		n2 = self._env.fromDevice(self._stats.getAverageDensity(cloud.b))

		N1 = n1.sum() * self._constants.dV
		N2 = n2.sum() * self._constants.dV
		N = N1 + N2

		n1 /= N1
		n2 /= N2

		chi = 1.0 / (2.0 * self._constants.hbar) * (
			self._constants.g[cloud.a.comp, cloud.a.comp] * (n1 * n1).sum() * self._constants.dV +
			self._constants.g[cloud.b.comp, cloud.b.comp] * (n2 * n2).sum() * self._constants.dV -
			2 * self._constants.g[cloud.a.comp, cloud.b.comp] * (n1 * n2).sum() * self._constants.dV)

		self.times.append(t)
		self.noise.append(numpy.sqrt(N) * chi * t)

	def getData(self):
		return [numpy.array(x) for x in (self.times, self.noise)]
