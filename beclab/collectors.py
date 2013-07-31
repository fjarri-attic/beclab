import numpy
import math

from .helpers import *
from .constants import REPR_CLASSICAL, REPR_WIGNER
from .meters import ProjectionMeter, UncertaintyMeter, IntegralMeter, getXiSquared, getSpins
from .evolution import TerminateEvolution
from .pulse import Pulse
from .wavefunction import WavefunctionSet


class ParticleNumberCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False, pulse=None):
		PairedCalculation.__init__(self, env)
		self.verbose = verbose
		self._pulse = pulse

		self.times = []
		self.N = []
		self.Nerr = []

		self._psi = WavefunctionSet(env, constants, grid)
		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._psi.prepare(components=self._p.components,
			ensembles=self._p.ensembles, psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		psi.copyTo(self._psi)

		if self._pulse is not None:
			self._pulse.apply(self._psi, theta=0.5 * numpy.pi)

		N = self._psi.density_meter.getNTotal()
		if self.verbose:
			print "Particle counter: ", t, "s,", N, N.sum()

		Ns = self._env.fromDevice(self._psi.density_meter.getNPerEnsemble())
		self.Nerr.append(Ns.std(1) / numpy.sqrt(float(Ns.shape[1])))

		self.times.append(t)
		self.N.append(N)

	def getData(self):
		N = numpy.array(self.N)
		Nerr = numpy.array(self.Nerr)
		return numpy.array(self.times), N.transpose(), N.sum(1), Nerr.transpose()


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
		self._unc = UncertaintyMeter(env, constants, grid)
		self.times = []
		self.phnoise = []
		self._verbose = verbose

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._unc.prepare(components=self._p.components, ensembles=self._p.ensembles,
			psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		phnoise = self._unc.getPhaseNoise(psi)

		if self._verbose:
			print "Phase noise:", t, "s,", phnoise

		self.times.append(t)
		self.phnoise.append(phnoise)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.phnoise)


class PzNoiseCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False):
		PairedCalculation.__init__(self, env)
		self._unc = UncertaintyMeter(env, constants, grid)
		self.times = []
		self.pznoise = []
		self._verbose = verbose

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._unc.prepare(components=self._p.components, ensembles=self._p.ensembles,
			psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		pznoise = self._unc.getPzNoise(psi)

		if self._verbose:
			print "Pz noise:", t, "s,", pznoise

		self.times.append(t)
		self.pznoise.append(pznoise)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.pznoise)


class VisibilityCollector(PairedCalculation):

	def __init__(self, env, constants, grid, verbose=False):
		PairedCalculation.__init__(self, env)
		self._int = IntegralMeter(env, constants, grid)
		self.verbose = verbose

		self.times = []
		self.visibility = []
		self.Verr = []

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._int.prepare(components=self._p.components,
			ensembles=self._p.ensembles, psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		v = self._int.getVisibility(psi)

		if self.verbose:
			print "Visibility: ", t, "s,", v

		Vs = self._int.getVisibilityPerEnsemble(psi)
		self.Verr.append(Vs.std() / numpy.sqrt(float(Vs.size)))
		self.times.append(t)
		self.visibility.append(v)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.visibility), numpy.array(self.Verr)


class SurfaceProjectionCollector(PairedCalculation):

	def __init__(self, env, constants, grid, pulse=None):
		PairedCalculation.__init__(self, env)
		self._projection = ProjectionMeter(env, constants, grid)
		self._pulse = pulse
		self._constants = constants
		self._grid = grid

		self._psi = WavefunctionSet(env, constants, grid)

		self.times = []
		self.xy = []
		self.yz = []
		self.xz = []

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
		self.xz.append(self._projection.getXZ(self._psi))

	def getData(self):
		shape = (len(self.times), self._p.components)
		shape_xy = shape + (self._grid.shape[1:3])
		shape_yz = shape + (self._grid.shape[:2])
		shape_xz = shape + (self._grid.shape[0], self._grid.shape[2])
		return numpy.array(self.times), \
			numpy.transpose(numpy.concatenate(self.xy).reshape(*shape_xy), axes=(1, 0, 2, 3)), \
			numpy.transpose(numpy.concatenate(self.yz).reshape(*shape_yz), axes=(1, 0, 2, 3)), \
			numpy.transpose(numpy.concatenate(self.xz).reshape(*shape_xz), axes=(1, 0, 2, 3))


class AxialProjectionCollector(PairedCalculation):

	def __init__(self, env, constants, grid, pulse=None):
		PairedCalculation.__init__(self, env)
		self._projection = ProjectionMeter(env, constants, grid)
		self._pulse = pulse
		self._constants = constants
		self._grid = grid

		self.times = []
		self.snapshots = []
		self.n1 = []
		self.n2 = []
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
		self.n1.append(proj[0])
		self.n2.append(proj[1])

	def getData(self):
		return numpy.array(self.times), \
			numpy.concatenate(self.snapshots).reshape(
				len(self.times), self.snapshots[0].size).transpose(), \
			numpy.concatenate(self.n1).reshape(
				len(self.times), self.snapshots[0].size).transpose(), \
			numpy.concatenate(self.n2).reshape(
				len(self.times), self.snapshots[0].size).transpose()


class WavefunctionCollector(PairedCalculation):

	def __init__(self, env, constants, grid, pulse=None):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self.times = []
		self.psis = []

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		pass

	def __call__(self, t, dt, psi):

		self.times.append(t)
		self.psis.append(psi.data.get())

	def getData(self):
		return numpy.array(self.times), self.psis


class UncertaintyCollector(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._unc = UncertaintyMeter(env, constants, grid)
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


class SpinCloudCollector(PairedCalculation):

	def __init__(self, env, constants, grid):
		PairedCalculation.__init__(self, env)
		self._unc = UncertaintyMeter(env, constants, grid)
		self.times = []
		self.phi = []
		self.yps = []
		self.Sx = []
		self.Sy = []
		self.Sz = []

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)

	def _prepare(self):
		self._unc.prepare(components=self._p.components, ensembles=self._p.ensembles,
			psi_type=self._p.psi_type)

	def __call__(self, t, dt, psi):
		self.times.append(t)
		i, n = self._unc.getEnsembleSums(psi)
		Si = [i.real, i.imag, 0.5 * (n[0] - n[1])]
		self.Sx.append(Si[0])
		self.Sy.append(Si[1])
		self.Sz.append(Si[2])
		phi, yps = getSpins(i, n[0], n[1])
		self.phi.append(phi)
		self.yps.append(yps)

	def getData(self):
		return [numpy.array(x) for x in (self.times, self.phi, self.yps,
			self.Sx, self.Sy, self.Sz)]


class AnalyticNoiseCollector:
	"""
	According to Ueda and Kitagawa, http://link.aps.org/doi/10.1103/PhysRevA.47.5138, (5)
	"""

	def __init__(self, env, constants, grid):
		self._env = env
		self._constants = constants
		self._grid = grid
		self.times = []
		self.noise = []

	def prepare(self, **kwds):
		pass

	def __call__(self, t, dt, psi):
		n = self._env.fromDevice(psi.density_meter.getNDensityAverage())
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
