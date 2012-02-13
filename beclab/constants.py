"""
Module, containing class with calculation constants
"""

import copy
import numpy

from .helpers import *
from .helpers.fht import getEigenfunction1D, getEigenfunction3D


# Physical constants
_R_BOHR = 5.2917720859e-11 # Bohr radius
_HBAR = 1.054571628e-34 # Planck constant

# representations
REPR_CLASSICAL = 0
REPR_WIGNER = 1


_DEFAULTS = {
	# scattering lengths for |1,-1> and |2,1>, in Bohr radii
	# source:
	# private communication with Servaas Kokkelmans and the paper
	# B. J. Verhaar, E. G. M. van Kempen, and S. J. J.
	# M. F. Kokkelmans, Phys. Rev. A 79, 032711 (2009).
	'a11': 100.4,
	'a22': 95.68,
	'a12': 98.13,

	# FIXME: basically the same as a**
	# Will be removed when keyword parsing is implemented
	'g11': None,
	'g12': None,
	'g22': None,

	# Spin-orbit coupling
	'lambda_R': 0,

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

	'e_cut': None,
}

_SO_2D_DEFAULTS = {
	# dimensionless parameters
	'g_ratio': 1, # g_interspecies / g_intraspecies
	'g_strength': 1, # g_intraspecies * N / (hbar * w_perp) / a_perp ** 2
	'lambda_SO': 1, # dimensionless coupling strength, a_perp / a_lambda

	# free experimental parameters
	'a': 100, # scattering length for intra-species interaction, Bohr radii
	'm': 1.443160648e-25, # mass of one particle (rubidium-87)
	'f_z': 1.9e3, # frequency of 2D-creating confinement, Hz
	'f_perp': 20, # frequency of the actual trap, Hz
}

_SO_1D_DEFAULTS = {
	# dimensionless parameters
	'g_ratio': 1, # g_interspecies / g_intraspecies
	'g_strength': 1, # g_intraspecies * N / (hbar * w_perp) / a_perp ** 2
	'lambda_SO': 1, # dimensionless coupling strength, a_perp / a_lambda

	# free experimental parameters
	'a': 100, # scattering length for intra-species interaction, Bohr radii
	'm': 1.443160648e-25, # mass of one particle (rubidium-87)
	'f_z': 1.9e3, # frequency of 2D-creating confinement, Hz
	'R': 1e-5, # confining radius, m
	'I': 1, # \int\limits_0^\inf F(r)^4 r dr, where F(r) is the ground state of ring potential
}


def buildPlaneWaveEnergy(constants, grid):
	"""
	Builds array with values of k-space energy
	(coefficients for kinetic term) in hbar units
	"""
	if grid.dim == 1:
		E = constants.hbar * grid.kz_full ** 2 / (2.0 * constants.m)
	elif grid.dim == 2:
		E = constants.hbar * \
			(grid.kx_full ** 2 +
			grid.ky_full ** 2) / (2.0 * constants.m)
	elif grid.dim == 3:
		E = constants.hbar * \
			(grid.kx_full ** 2 +
			grid.ky_full ** 2 +
			grid.kz_full ** 2) / (2.0 * constants.m)

	return E.astype(constants.scalar.dtype)

def buildPlaneWaveDensityModifiers(constants, grid):
	"""
	Builds array with modifiers for calculating density in Wigner representation
	(values of 0.5 \delta_P(x, x) on the grid)
	"""
	mask = buildProjectorMask(constants, grid)
	modifiers = numpy.ones(grid.shape) * mask.sum() / (2.0 * grid.V)
	return modifiers.astype(constants.scalar.dtype)

def buildHarmonicEnergy(constants, grid):
	"""
	Builds array with energy values in harmonic mode space in hbar units
	"""

	if grid.dim == 3:
		mx, my, mz = grid.mx_full, grid.my_full, grid.mz_full
		wx, wy, wz = constants.wx, constants.wy, constants.wz
		E = (wx * (mx + 0.5) + wy * (my + 0.5) + wz * (mz + 0.5))
	elif grid.dim == 1:
		E = (constants.wz * (grid.mz_full + 0.5))

	return E.astype(constants.scalar.dtype)

def buildHarmonicDensityModifiers(constants, grid):
	"""
	Builds array with modifiers for calculating density in Wigner representation
	(values of 0.5 \delta_P(x, x) on the grid)
	"""
	mask = buildProjectorMask(constants, grid)
	modifiers = numpy.zeros(grid.shape)

	if grid.dim == 3:
		x, y, z = grid.x_full / grid.lx, grid.y_full / grid.ly, grid.z_full / grid.lz
		coeff = 0.5 / (grid.lx * grid.ly * grid.lz)
		for ix in xrange(grid.shape[2]):
			for iy in xrange(grid.shape[1]):
				for iz in xrange(grid.shape[0]):
					if mask[iz, iy, ix] == 0: continue
					eigenfunc = getEigenfunction3D(ix, iy, iz)
					modifiers += eigenfunc(x, y, z) ** 2
	elif grid.dim == 1:
		z = grid.z_full / grid.lz
		coeff = 0.5 / grid.lz
		for iz in xrange(grid.shape[0]):
			if mask[iz] == 0: continue
			eigenfunc = getEigenfunction1D(iz)
			modifiers += eigenfunc(z) ** 2

	modifiers *= coeff
	return modifiers.astype(constants.scalar.dtype)

def buildPotentials(constants, grid):
	"""
	Builds array with values of external potential energy (in hbar units).
	"""

	if grid.dim == 1:
		z = grid.z_full

		potentials = constants.m * ((constants.wz * z) ** 2) / (2.0 * constants.hbar)

	elif grid.dim == 2:
		x, y = grid.x_full, grid.y_full

		potentials = constants.m * (
			(constants.wx * x) ** 2 +
			(constants.wy * y) ** 2
		) / (2.0 * constants.hbar)

	elif grid.dim == 3:
		x, y, z = grid.x_full, grid.y_full, grid.z_full

		potentials = constants.m * (
			(constants.wx * x) ** 2 +
			(constants.wy * y) ** 2 +
			(constants.wz * z) ** 2) / (2.0 * constants.hbar)

	return potentials.astype(constants.scalar.dtype).reshape((1,) + potentials.shape)

def buildSOEnergy(constants, grid):
	if grid.dim == 1:
		return NotImplementedError()
	if grid.dim == 2:
		kx, ky = grid.kx_full, grid.ky_full
		k2 = kx ** 2 + ky ** 2
		diff2 = constants.hbar * k2 / 2 / constants.m
		int1 = constants.lambda_R / constants.hbar * (ky + 1j * kx)
		int2 = constants.lambda_R / constants.hbar * (ky - 1j * kx)
		return numpy.array([
			[diff2, int1], [int2, diff2]
		]).astype(constants.complex.dtype)
	else:
		raise NotImplementedError()

def buildEnergyExp(energy, dt=1, imaginary_time=False):
	e = energy * (-1j * dt * (-1j if imaginary_time else 1))
	return elementwiseMatrixExp(e)

def buildProjectorMask(constants, grid):
	"""
	Returns array in mode space with 1.0 at places where mode energy is smaller
	than cutoff energy and 0.0 everywhere else.
	"""

	if isinstance(grid, UniformGrid):
		E = buildPlaneWaveEnergy(constants, grid)
	else:
		E = buildHarmonicEnergy(constants, grid)

	if constants.e_cut is not None:
		mask_func = lambda x: 0.0 if x > constants.e_cut * constants.hbar else 1.0
		mask_map = numpy.vectorize(mask_func)
		mask = mask_map(E * constants.hbar).astype(constants.scalar.dtype)
	else:
		mask = numpy.ones_like(E)

	return mask.astype(constants.scalar.dtype)


def _buildIntegrationCoefficients1D(pts):
	"""
	Returns integration coefficients for simple trapezoidal rule.
	Works for non-uniform grids.
	"""
	return numpy.array(
		[(pts[1] - pts[0]) / 2] +
		((pts[2:] - pts[:-2]) / 2.0).tolist() +
		[(pts[-1] - pts[-2]) / 2])

def buildIntegrationCoefficients(constants, grid):
	if grid.dim == 1:
		dV = grid.dz
	elif grid.dim == 2:
		dx, dy = tile2D(grid.dx, grid.dy)
		dV = dx * dy
	elif grid.dim == 3:
		dx, dy, dz = tile3D(grid.dx, grid.dy, grid.dz)
		dV = dx * dy * dz

	return dV.astype(constants.scalar.dtype)


class GridBase:

	# these arrays will be created dynamically on first request
	# (their creation takes some time, and in case of large grids they
	# need significant amount of memory)
	__data_arrays__ = {}

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants

	def __getattr__(self, name):
		if name.endswith('_device'):
			array_name = name[:-7]
			need_on_device = True
		else:
			array_name = name
			need_on_device = False

		if array_name not in self.__data_arrays__:
			raise AttributeError(name)

		# create array on CPU if it does not exist yet
		if array_name not in self.__dict__:
			build_func = self.__data_arrays__[array_name]
			setattr(self, array_name, build_func(self._constants, self))

		# transfer array to device if necessary
		if need_on_device:
			array = getattr(self, array_name)
			setattr(self, name, self._env.toDevice(array))

		return getattr(self, name)


class UniformGrid(GridBase):

	def __init__(self, env, constants, shape, box_size):

		GridBase.__init__(self, env, constants)

		assert (isinstance(shape, int) and isinstance(size, float)) or \
			(len(shape) == len(box_size))

		self.__data_arrays__ = {
			'energy': buildPlaneWaveEnergy if constants.lambda_R == 0 else buildSOEnergy,
			'density_modifiers': buildPlaneWaveDensityModifiers,
			'potentials': buildPotentials,
			'dV': buildIntegrationCoefficients,
			'projector_mask': buildProjectorMask,
		}

		if isinstance(shape, int):
			shape = (shape,)
			box_size = (box_size,)

		assert len(shape) in [1, 2, 3]

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
		self.dV_uniform = 1.0
		self.size = 1
		for i in xrange(self.dim):
			self.dV_uniform *= d_space[i]
			self.size *= self.shape[i]
		self.msize = self.size

		self.V = self.dV_uniform * self.size

		kvalues = lambda dx, N: numpy.fft.fftfreq(N, dx) * 2.0 * numpy.pi

		if self.dim == 3:

			# 1D grids
			self.x = grid_space[2]
			self.y = grid_space[1]
			self.z = grid_space[0]

			# tiled grid arrays to use in elementwise numpy operations
			self.x_full, self.y_full, self.z_full = tile3D(self.x, self.y, self.z)

			self.kx = kvalues(d_space[2], self.shape[2])
			self.ky = kvalues(d_space[1], self.shape[1])
			self.kz = kvalues(d_space[0], self.shape[0])

			self.kx_full, self.ky_full, self.kz_full = tile3D(self.kx, self.ky, self.kz)

			# coefficients for integration
			self.dx = _buildIntegrationCoefficients1D(self.x)
			self.dy = _buildIntegrationCoefficients1D(self.y)
			self.dz = _buildIntegrationCoefficients1D(self.z)

		elif self.dim == 2:

			# 1D grids
			self.x = grid_space[1]
			self.y = grid_space[0]

			# tiled grid arrays to use in elementwise numpy operations
			self.x_full, self.y_full = tile2D(self.x, self.y)

			self.kx = kvalues(d_space[1], self.shape[1])
			self.ky = kvalues(d_space[0], self.shape[0])

			self.kx_full, self.ky_full = tile2D(self.kx, self.ky)

			# coefficients for integration
			self.dx = _buildIntegrationCoefficients1D(self.x)
			self.dy = _buildIntegrationCoefficients1D(self.y)

		else:
			# using 'z' axis for 1D, because it seems more natural
			self.z = grid_space[0]
			self.z_full = self.z

			self.kz = kvalues(d_space[0], self.shape[0])
			self.kz_full = self.kz

			self.dz = _buildIntegrationCoefficients1D(self.z)

	@classmethod
	def forN(cls, env, constants, N, shape, border=1.2):
		"""Create suitable lattice for trapped cloud of N atoms"""
		box_size = constants.boxSizeForN(N, len(shape), border=border)
		return cls(env, constants, shape, box_size)


class HarmonicGrid(GridBase):

	__data_arrays__ = {
		'energy': buildHarmonicEnergy,
		'density_modifiers': buildHarmonicDensityModifiers,
		'potentials': buildPotentials,
		'dV': buildIntegrationCoefficients,
		'projector_mask': buildProjectorMask,
	}

	def __init__(self, env, constants, mshape):

		GridBase.__init__(self, env, constants)

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
				self.dxs[l] = _buildIntegrationCoefficients1D(self.xs[l])
				self.dys[l] = _buildIntegrationCoefficients1D(self.ys[l])
				self.dzs[l] = _buildIntegrationCoefficients1D(self.zs[l])

				dx, dy, dz = tile3D(self.dxs[l], self.dys[l], self.dzs[l])
				self.dVs[l] = dx * dy * dz

			else:
				self.zs[l], _ = getHarmonicGrid(mshape[0], l)
				self.zs[l] *= self.lz

				self.shapes[l] = (len(self.zs[l]),)

				self.zs_full[l] = self.zs[l]

				self.dzs[l] = _buildIntegrationCoefficients1D(self.zs[l])
				self.dVs[l] = self.dzs[l]

		# Create aliases for 1st order arrays,
		# making it look like UniformGrid
		# (high orders are used only inside evolution classes anyway)
		self.shape = self.shapes[1]

		self.sizes = {}
		for j in (1, 2, 3, 4):
			self.sizes[j] = 1
			for i in xrange(self.dim):
				self.sizes[j] *= self.shapes[j][i]
		self.size = self.sizes[1]

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


class Constants:

	hbar = _HBAR
	r_bohr = _R_BOHR

	def __init__(self, double=True, use_effective_area=False, **kwds):

		self.double = double
		precision = double_precision if double else single_precision
		self.scalar = precision.scalar
		self.complex = precision.complex

		self.__dict__.update(_DEFAULTS)

		for key in kwds.keys():
			assert key in _DEFAULTS
		self.__dict__.update(kwds)

		self.so_coupling = (self.lambda_R != 0)

		self.wx = 2.0 * numpy.pi * self.fx
		self.wy = 2.0 * numpy.pi * self.fy
		self.wz = 2.0 * numpy.pi * self.fz

		if self.g11 is None:
			g = lambda a: 4.0 * numpy.pi * (self.hbar ** 2) * a * self.r_bohr / self.m
			self.g = numpy.array([
				[g(self.a11), g(self.a12)],
				[g(self.a12), g(self.a22)]
			])
		else:
			self.g = numpy.array([
				[self.g11, self.g12],
				[self.g12, self.g22]
			])

		# TODO: need to implement some mechanism of passing this information
		# from Constants constructor (parsing names of keyword arguments?)
		l = {
			(1, 1): self.gamma12 / 2,
			(0, 2): self.gamma22 / 4,
			(3, 0): self.gamma111 / 6
		}

		if use_effective_area:
			S = self.getEffectiveArea()
			self.g /= S

			for components in l:
				l[components] /= S ** (numpy.array(components).sum() - 1)

		# Fill special arrays for drift and diffusion terms in GPE

		self.losses_drift = [] # 'Gammas'
		self.losses_diffusion = [] # 'betas'

		for comp in xrange(2):
			drift = []
			diffusion = []

			for orders, kappa in l.iteritems():
				# appending empty diffusion term in order to account for
				# different noise sources for different losses
				if orders[comp] == 0:
					diffusion.append([0, ()])
				else:
					drift_orders = list(orders)
					drift_coeff = orders[comp] * kappa
					drift_orders[comp] -= 1

					diffusion_orders = list(orders)
					diffusion_orders[comp] -= 1
					diffusion_coeff = orders[comp] * numpy.sqrt(kappa)

					drift.append([drift_coeff, drift_orders])
					diffusion.append([diffusion_coeff, diffusion_orders])

			self.losses_drift.append(drift)
			self.losses_diffusion.append(diffusion)

		self.noise_sources = len(l)

	def getEffectiveArea(self):
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
			((self.m * self.wz * self.wz / 2) ** (1.0 / 3))

	def boxSizeForN(self, N, dim, comp=0, border=1.2):
		"""
		Returns size of bounding box which contains cloud with given parameters
		"""

		# calculating approximate diameter of the cloud based on
		# Thomas-Fermi chemical potential
		mu = self.muTF(N, dim=dim, comp=comp)
		diameter = lambda w: 2.0 * border * numpy.sqrt(2.0 * mu / (self.m * w ** 2))

		if dim == 3:
			box_size = (diameter(self.wz), diameter(self.wy), diameter(self.wx))
		else:
			box_size = (diameter(self.wz),)

		return box_size

	def harmonicModesForCutoff(self, dim):
		"""
		Returns minimal grid size which contains all possible modes with
		energy below cutoff.
		"""
		assert self.e_cut is not None

		if dim == 3:
			wx, wy, wz = self.wx, self.wy, self.wz
			xmodes = int((self.e_cut - wy / 2 - wz / 2) / wx + 0.5)
			ymodes = int((self.e_cut - wx / 2 - wz / 2) / wy + 0.5)
			zmodes = int((self.e_cut - wx / 2 - wy / 2) / wz + 0.5)
			return (zmodes, ymodes, xmodes)
		elif dim == 1:
			zmodes = int(self.e_cut / self.wz + 0.5)
			return (zmodes,)

	def planeWaveModesForCutoff(self, box_size):
		"""
		Returns minimal grid size which contains all possible modes with
		energy below cutoff.
		"""
		assert self.e_cut is not None
		dim = len(box_size)

		def get_modes(size):
			ncut = size / (2 * numpy.pi) * numpy.sqrt(2 * self.m * self.e_cut / self.hbar)

			# formula taken from numpy.fft.fftfreq
			nk = lambda N: float((N / 2) * (N - 1)) / N

			# find limiting even N (they have simple form of nk(N) function)
			Neven = 2 * int((2 * ncut + 1) / 2)

			# check that next odd nk is still lower than ncut
			Nodd = Neven + 1
			if nk(Nodd) < ncut:
				return Nodd
			else:
				return Neven

		if dim == 3:
			zsize, ysize, xsize = box_size
			return(get_modes(zsize), get_modes(ysize), get_modes(xsize))
		elif dim == 1:
			return (get_modes(box_size[0]),)


class SOConstants2D:

	def __init__(self, **kwds):

		self.__dict__.update(_SO_2D_DEFAULTS)
		for key in kwds.keys():
			assert key in _SO_2D_DEFAULTS
		self.__dict__.update(kwds)

		w_z = self.f_z * numpy.pi * 2
		w_perp = self.f_perp * numpy.pi * 2

		a_z = numpy.sqrt(_HBAR / (self.m * w_z))
		a_perp = numpy.sqrt(_HBAR / (self.m * w_perp)) # characteristic length

		g = numpy.sqrt(numpy.pi * 8) * (_HBAR ** 2 / self.m) * (self.a * _R_BOHR / a_z)

		N = self.g_strength * _HBAR * w_perp * a_perp ** 2 / g # + 1

		a_lambda = a_perp / self.lambda_SO
		lambda_R = _HBAR ** 2 / (self.m * a_lambda)

		# save calculated parameters
		self.a_perp = a_perp
		self.g_intra = g
		self.g_inter = g * self.g_ratio
		self.lambda_R = lambda_R
		self.N = N


class SOConstants1D:

	def __init__(self, **kwds):

		self.__dict__.update(_SO_1D_DEFAULTS)
		for key in kwds.keys():
			assert key in _SO_1D_DEFAULTS
		self.__dict__.update(kwds)

		w_z = self.f_z * numpy.pi * 2
		a_z = numpy.sqrt(_HBAR / (self.m * w_z))

		g_2D = numpy.sqrt(numpy.pi * 8) * (_HBAR ** 2 / self.m) * (self.a * _R_BOHR / a_z)
		g = g_2D * self.m * self.R ** 2 / _HBAR ** 2 * self._I

		N = self.g_strength / g # + 1

		a_lambda = a_perp / self.lambda_SO
		lambda_R = self._lambda_SO * _HBAR ** 2 / (self.m * self.R)

		# save calculated parameters
		self.box_size = 2 * numpy.pi * self.R
		self.g_intra = g
		self.g_inter = g * self.g_ratio
		self.lambda_R = lambda_R
		self.N = N
