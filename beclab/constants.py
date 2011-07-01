"""
Module, containing class with calculation constants
"""

import copy
import numpy

from .helpers import *

# representations
CLASSICAL = 0
WIGNER = 1

_DEFAULTS = {
	# scattering lengths for |1,-1> and |2,1>, in Bohr radii
	# source:
	# private communication with Servaas Kokkelmans and the paper
	# B. J. Verhaar, E. G. M. van Kempen, and S. J. J.
	# M. F. Kokkelmans, Phys. Rev. A 79, 032711 (2009).
	'a11': 100.4,
	'a22': 95.68,
	'a12': 98.13,

	# mass of one particle (rubidium-87)
	'm': 1.443160648e-25,

	# Trap frequencies
	'fx': 97.6,
	'fy': 97.6,
	'fz': 11.96,

	# loss terms (according to M. Egorov, as of 28 Feb 2011; for 44k atoms)
	'gamma111': 5.4e-42,
	'gamma12': 1.52e-20,
	'gamma22': 7.7e-20,

	# number of iterations for mid-step of split-step evolution
	'itmax': 3,
}

def getPotentials(env, constants, grid):
	"""Returns array with values of external potential energy (in hbar units)."""

	if grid.dim == 1:
		z = grid.z_full

		potentials = constants.m * ((constants.wz * z) ** 2) / (2.0 * constants.hbar)
	else:
		x, y, z = grid.x_full, grid.y_full, grid.z_full

		potentials = constants.m * (
			(constants.wx * x) ** 2 +
			(constants.wy * y) ** 2 +
			(constants.wz * z) ** 2) / (2.0 * constants.hbar)

	potentials = potentials.astype(constants.scalar.dtype)

	if env is not None:
		return env.toDevice(potentials)
	else:
		return potentials

def getPlaneWaveEnergy(env, constants, grid):
	"""
	Returns array with values of k-space energy
	(coefficients for kinetic term) in hbar units
	"""
	assert isinstance(grid, UniformGrid)

	if grid.dim == 1:
		E = constants.hbar * grid.kz_full ** 2 / (2.0 * constants.m)
	else:
		E = constants.hbar * \
			(grid.kx_full ** 2 + grid.ky_full ** 2 + grid.kz_full ** 2) / (2.0 * constants.m)

	E = E.astype(constants.scalar.dtype)

	if env is not None:
		return env.toDevice(E)
	else:
		return E

def getHarmonicEnergy(env, constants, grid):
	"""
	Returns array with energy values in harmonic mode space in hbar units
	"""

	assert isinstance(grid, HarmonicGrid)

	if grid.dim == 3:
		mx, my, mz = grid.mx_full, grid.my_full, grid.mz_full
		wx, wy, wz = constants.wx, constants.wy, constants.wz
		E = (wx * (mx + 0.5) + wy * (my + 0.5) + wz * (mz + 0.5))
	else:
		E = (constants.wz * (grid.mz_full + 0.5))

	E = E.astype(constants.scalar.dtype)

	if env is None:
		return E
	else:
		return env.toDevice(E)

def getProjectorMask(env, constants, grid):
	if isinstance(grid, UniformGrid):
		E = getPlaneWaveEnergy(None, constants, grid)
	else:
		E = getHarmonicEnergy(None, constants, grid)

	mask_func = lambda x: 0.0 if x > constants.e_cut else 1.0
	mask_map = numpy.vectorize(mask_func)

	mask = mask_map(E * constants.hbar).astype(constants.scalar.dtype)
	modes = numpy.sum(mask)

	return env.toDevice(mask)


class UniformGrid:

	def __init__(self, env, constants, shape, box_size):

		self._constants = constants.copy()

		assert (isinstance(shape, int) and isinstance(size, float)) or \
			(len(shape) == len(box_size))

		if isinstance(shape, int):
			shape = (shape,)
			box_size = (box_size,)

		assert len(shape) in [1, 3]

		self.dim = len(shape)
		self.shape = shape
		self.mshape = shape

		# spatial step and grid for every component of shape
		d_space = [box_size[i] / (shape[i] - 1) for i in xrange(self.dim)]
		grid_space = [
			-box_size[i] / 2.0 + d_space[i] * numpy.arange(shape[i])
			for i in xrange(self.dim)
		]

		# number of cells and cell volume
		dV = 1.0
		self.size = 1
		for i in xrange(self.dim):
			dV *= d_space[i]
			self.size *= self.shape[i]
		self.msize = self.size

		self.V = dV * self.size

		kvalues = lambda dx, N: numpy.fft.fftfreq(N, dx) * 2.0 * numpy.pi

		if self.dim == 3:
			self.dx = d_space[2]
			self.dy = d_space[1]
			self.dz = d_space[0]

			# 1D grids
			self.x = grid_space[2]
			self.y = grid_space[1]
			self.z = grid_space[0]

			# tiled grid arrays to use in elementwise numpy operations
			self.x_full, self.y_full, self.z_full = tile3D(self.x, self.y, self.z)

			self.kx = kvalues(self.dx, self.shape[2])
			self.ky = kvalues(self.dy, self.shape[1])
			self.kz = kvalues(self.dz, self.shape[0])

			self.kx_full, self.ky_full, self.kz_full = tile3D(self.kx, self.ky, self.kz)

			# coefficients for integration;
			# multiplying border coefficients by 0.5, according to simple trapezoidal rule
			# (although function is zero there anyway, so it doesn't really matter)
			dx = numpy.array([0.5] + [1.0] * (self.shape[2] - 2) + [0.5])
			dy = numpy.array([0.5] + [1.0] * (self.shape[1] - 2) + [0.5])
			dz = numpy.array([0.5] + [1.0] * (self.shape[0] - 2) + [0.5])
			dx, dy, dz = tile3D(dx, dy, dz)
			self.dV = dx * dy * dz * dV

		else:
			# using 'z' axis for 1D, because it seems more natural
			self.dz = d_space[0]
			self.z = grid_space[0]
			self.z_full = self.z

			self.kz = kvalues(self.dz, self.shape[0])
			self.kz_full = self.kz

			dz = numpy.array([0.5] + [1.0] * (self.shape[0] - 2) + [0.5])
			self.dV = dz * dV

	@classmethod
	def forN(cls, env, constants, N, shape, border=1.2):
		"""Create suitable lattice for trapped cloud of N atoms"""

		# calculating approximate diameter of the cloud based on
		# Thomas-Fermi chemical potential for the first component
		dim = len(shape)
		mu1 = constants.muTF(N, dim=dim, comp=1)
		diameter = lambda w: 2.0 * border * numpy.sqrt(2.0 * mu1 / (constants.m * w ** 2))

		if dim == 3:
			box_size = (diameter(constants.wz), diameter(constants.wy), diameter(constants.wx))
		else:
			box_size = (diameter(constants.wz),)

		return cls(env, constants, shape, box_size)

	def copy(self):
		return copy.deepcopy(self)

	def get_dV(self, env):
		return env.toDevice(self.dV.astype(self._constants.scalar.dtype))

	def __eq__(self, other):
		return self.shape == other.shape and self.size == other.size


class HarmonicGrid:

	def __init__(self, env, constants, mshape):

		self._constants = constants.copy()

		if isinstance(mshape, int):
			mshape = (mshape,)

		assert len(mshape) in [1, 3]

		self.dim = len(mshape)
		self.mshape = mshape

		if self.dim == 3:
			mx = numpy.arange(mshape[2])
			my = numpy.arange(mshape[1])
			mz = numpy.arange(mshape[0])
			self.mx_full, self.my_full, self.mz_full = tile3D(mx, my, mz)
		else:
			self.mz_full = numpy.arange(mshape[0])

		self.msize = 1
		for i in xrange(self.dim):
			self.msize *= mshape[i]

		if self.dim == 3:
			wx = constants.wx
			wy = constants.wy
		wz = constants.wz

		# natural lengths for harmonic oscillator
		if self.dim == 3:
			self.lx = numpy.sqrt(constants.hbar / (wx * constants.m))
			self.ly = numpy.sqrt(constants.hbar / (wy * constants.m))
		self.lz = numpy.sqrt(constants.hbar / (wz * constants.m))

		# Spatial grids for collectors which work in x-space
		# and for nonlinear terms in GPEs
		# We have to create a set of grid for every transformation order used
		self.shapes = {}
		self.dVs = {}

		if self.dim == 3:
			self.xs = {}
			self.ys = {}
			self.zs = {}
			self.xs_full = {}
			self.ys_full = {}
			self.zs_full = {}
			self.dxs = {}
			self.dys = {}
			self.dzs = {}
		else:
			self.zs = {}
			self.zs_full = {}
			self.dzs = {}

		# Build coefficients for integration in x-space using simple trapezoidal rule
		ds = lambda pts: numpy.array(
			[(pts[1] - pts[0]) / 2] +
			((pts[2:] - pts[:-2]) / 2.0).tolist() +
			[(pts[-1] - pts[-2]) / 2])

		for l in (1, 2, 3, 4):
			if self.dim == 3:
				# non-uniform grid used in Gauss-Hermite quadrature
				self.xs[l], _ = getHarmonicGrid(mshape[2], l)
				self.ys[l], _ = getHarmonicGrid(mshape[1], l)
				self.zs[l], _ = getHarmonicGrid(mshape[0], l)

				self.shapes[l] = (len(self.zs[l]), len(self.ys[l]), len(self.xs[l]))

				self.xs[l] *= self.lx
				self.ys[l] *= self.ly
				self.zs[l] *= self.lz

				# tiled grid arrays to use in elementwise numpy operations
				self.xs_full[l], self.ys_full[l], self.zs_full[l] = \
					tile3D(self.xs[l], self.ys[l], self.zs[l])

				# Coefficients for integration
				self.dxs[l] = ds(self.xs[l])
				self.dys[l] = ds(self.ys[l])
				self.dzs[l] = ds(self.zs[l])

				dx, dy, dz = tile3D(self.dxs[l], self.dys[l], self.dzs[l])
				self.dVs[l] = dx * dy * dz

			else:
				self.zs[l], _ = getHarmonicGrid(mshape[0], l)
				self.zs[l] *= self.lz

				self.shapes[l] = (len(self.zs[l]),)

				self.zs_full[l] = self.zs[l]

				# dVs for debugging (integration in x-space)
				self.dzs[l] = ds(self.zs[l])
				self.dVs[l] = self.dzs[l]

			# Create aliases for 1st order arrays,
			# making it look like UniformGrid
			# (high orders are used only inside evolution classes anyway)
			self.shape = self.shapes[1]
			self.dV = self.dVs[1]

			self.size = 1
			for i in xrange(self.dim):
				self.size *= self.shape[i]

			if self.dim == 3:
				self.x = self.xs[1]
				self.y = self.ys[1]
				self.x_full = self.xs_full[1]
				self.y_full = self.ys_full[1]
				self.dx = self.dxs[1]
				self.dy = self.dys[1]

			self.z = self.zs[1]
			self.z_full = self.zs_full[1]
			self.dz = self.dzs[1]

	def copy(self):
		return copy.deepcopy(self)

	def get_dV(self, env, order=1):
		return env.toDevice(self.dVs[order].astype(self._constants.scalar.dtype))


class Constants:

	hbar = 1.054571628e-34 # Planck constant
	r_bohr = 5.2917720859e-11 # Bohr radius

	def __init__(self, double=True, **kwds):

		self.double = double
		precision = double_precision if double else single_precision
		self.scalar = precision.scalar
		self.complex = precision.complex

		self.__dict__.update(_DEFAULTS)

		for key in kwds.keys():
			assert key in _DEFAULTS
		self.__dict__.update(kwds)

		self.wx = 2.0 * numpy.pi * self.fx
		self.wy = 2.0 * numpy.pi * self.fy
		self.wz = 2.0 * numpy.pi * self.fz

		g11 = 4.0 * numpy.pi * (self.hbar ** 2) * self.a11 * self.r_bohr / self.m
		g12 = 4.0 * numpy.pi * (self.hbar ** 2) * self.a12 * self.r_bohr / self.m
		g22 = 4.0 * numpy.pi * (self.hbar ** 2) * self.a22 * self.r_bohr / self.m
		self.g = numpy.array([[g11, g12], [g12, g22]])

	def getEffectiveArea(self, grid):
		if grid.dim == 3:
			return 1.0

		l_rho = numpy.sqrt(self.hbar / (2.0 * self.m * self.wx))
		return 4.0 * numpy.pi * (l_rho ** 2)

	def muTF(self, N, dim=3, comp=0):
		g = self.g[comp, comp]

		if dim == 1:
			return self._muTF1D(N, g)
		else:
			return self._muTF3D(N, g)

	def _muTF3D(self, N, g):
		"""get TF-approximated chemical potential"""
		w = (self.wx * self.wy * self.wz) ** (1.0 / 3)
		return ((15 * N / (8.0 * numpy.pi)) ** 0.4) * \
			((self.m * w * w / 2) ** 0.6) * \
			(g ** 0.4)

	def _muTF1D(self, N, g):
		"""get TF-approximated chemical potential"""
		return ((0.75 * g * N) ** (2.0 / 3)) * \
			((self.m * self.wx * self.wx / 2) ** (1.0 / 3))

	def __eq__(self, other):
		for key in _DEFAULTS.keys() + ['double']:
			if getattr(self, key) != getattr(other, key):
				return False

		return True

	def copy(self):
		return copy.deepcopy(self)
