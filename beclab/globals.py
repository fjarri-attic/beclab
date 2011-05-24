"""
Auxiliary functions and classes.
"""

import numpy
import math


class PairedCalculation:
	"""
	Base class for paired GPU/CPU calculations.
	Depending on initializing parameter, it will make visible either _gpu_
	or _cpu_ methods.
	"""

	def __init__(self, env):
		self.__gpu = env.gpu
		self.__createAliases()

	def __findPrefixedMethods(self):
		if self.__gpu:
			prefix = "_gpu_"
		else:
			prefix = "_cpu_"

		res = {}
		for attr in dir(self):
			if attr.startswith(prefix):
				res[attr] = attr[len(prefix):]

		return res

	def __createAliases(self):
		to_add = self.__findPrefixedMethods()
		for attr in to_add:
			self.__dict__[to_add[attr]] = getattr(self, attr)

	def __deleteAliases(self, d):
		to_del = self.__findPrefixedMethods()
		for attr in to_del:
			del d[to_del[attr]]

	def __getstate__(self):
		d = dict(self.__dict__)
		self.__deleteAliases(d)
		return d

	def __setstate__(self, state):
		self.__dict__ = state
		self.__createAliases()


def log2(x):
	"""Calculates binary logarithm for integer"""
	pows = [1]
	while x > 2 ** pows[-1]:
		pows.append(pows[-1] * 2)

	res = 0
	for pow in reversed(pows):
		if x >= (2 ** pow):
			x >>= pow
			res += pow
	return res

def tile3D(x, y, z):
	nx = len(x)
	ny = len(y)
	nz = len(z)

	xx = numpy.tile(x, ny * nz).reshape(nz, ny, nx)
	yy = numpy.transpose(numpy.tile(y, nx * nz).reshape(nz, nx, ny), axes=(0, 2, 1))
	zz = numpy.transpose(numpy.tile(z, ny * nx).reshape(nx, ny, nz), axes=(2, 1, 0))

	return xx, yy, zz

def getPotentials(env, constants):
	"""Returns array with values of external potential energy (in hbar units)."""

	potentials = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	if constants.dim == 1:
		z = -constants.zmax + numpy.arange(constants.nvz) * constants.dz
		potentials = constants.m * ((constants.w_z * z) ** 2) / (2.0 * constants.hbar)

	else:
		i, j, k = tile3D(numpy.arange(constants.nvx), numpy.arange(constants.nvy),
			numpy.arange(constants.nvz))

		x = -constants.xmax + i * constants.dx
		y = -constants.ymax + j * constants.dy
		z = -constants.zmax + k * constants.dz

		potentials = constants.m * (
			(constants.w_x * x) ** 2 +
			(constants.w_y * y) ** 2 +
			(constants.w_z * z) ** 2) / (2.0 * constants.hbar)

	return env.toDevice(potentials.astype(constants.scalar.dtype))

def getKVectors(env, constants):
	"""
	Returns array with values of k-space vectors
	(coefficients for kinetic term) in hbar units
	"""
	def kvalues(dx, N):
		return numpy.fft.fftfreq(N, dx) * 2.0 * math.pi

	kvectors = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	if constants.dim == 1:
		kz = kvalues(constants.dz, constants.nvz)
		kvectors = constants.hbar * kz ** 2 / (2.0 * constants.m)

	else:
		kx = kvalues(constants.dx, constants.nvx)
		ky = kvalues(constants.dy, constants.nvy)
		kz = kvalues(constants.dz, constants.nvz)
		kx, ky, kz = tile3D(kx, ky, kz)

		kvectors = constants.hbar * \
			(kx * kx + ky * ky + kz * kz) / (2.0 * constants.m)

	return env.toDevice(kvectors.astype(constants.scalar.dtype))


def getProjectorArray(constants):

	def kvalues(dx, N):
		return numpy.fft.fftfreq(N, dx) * 2.0 * math.pi

	mask = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	mask_func = lambda x: 0.0 if x > constants.e_cut else 1.0
	mask_map = numpy.vectorize(mask_func)

	if constants.dim == 1:
		kz = kvalues(constants.dz, constants.nvz)
		e_k = (constants.hbar * kz) ** 2 / (2.0 * constants.m)
		mask = mask_map(e_k)

	else:
		kx = kvalues(constants.dx, constants.nvx)
		ky = kvalues(constants.dy, constants.nvy)
		kz = kvalues(constants.dz, constants.nvz)
		kx, ky, kz = tile3D(kx, ky, kz)

		e_k = (constants.hbar ** 2) * \
			(kx * kx + ky * ky + kz * kz) / (2.0 * constants.m)

		mask = mask_map(e_k)

	modes = numpy.sum(mask)
	#print "Projector modes: " + str(modes) + " out of " + str(constants.cells)

	return mask.astype(constants.scalar.dtype), modes

def getProjectorMask(env, constants):
	mask, _ = getProjectorArray(constants)
	return env.toDevice(mask)