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

def getPotentials(env, constants):
	"""Returns array with values of external potential energy (in hbar units)."""

	potentials = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	if constants.dim == 1:
		for k in xrange(constants.nvz):
			z = -constants.zmax + k * constants.dz

			potentials[k] = constants.m * (
				(constants.w_z * z) ** 2) / (2.0 * constants.hbar)

	else:
		for i in xrange(constants.nvx):
			for j in xrange(constants.nvy):
				for k in xrange(constants.nvz):
					x = -constants.xmax + i * constants.dx
					y = -constants.ymax + j * constants.dy
					z = -constants.zmax + k * constants.dz

					potentials[k, j, i] = constants.m * (
						(constants.w_x * x) ** 2 +
						(constants.w_y * y) ** 2 +
						(constants.w_z * z) ** 2) / (2.0 * constants.hbar)

	return env.toDevice(potentials)

def getKVectors(env, constants):
	"""
	Returns array with values of k-space vectors
	(coefficients for kinetic term) in hbar units
	"""
	def kvalues(dx, N):
		return numpy.fft.fftfreq(N, dx) * 2.0 * math.pi

	kvectors = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	# FIXME: use elementwise operations
	if constants.dim == 1:
		for k, kz in enumerate(kvalues(constants.dz, constants.nvz)):
			kvectors[k] = constants.hbar * kz * kz / (2.0 * constants.m)

	else:
		for i, kx in enumerate(kvalues(constants.dx, constants.nvx)):
			for j, ky in enumerate(kvalues(constants.dy, constants.nvy)):
				for k, kz in enumerate(kvalues(constants.dz, constants.nvz)):

					kvectors[k, j, i] = constants.hbar * \
						(kx * kx + ky * ky + kz * kz) / (2.0 * constants.m)

	return env.toDevice(kvectors)

def getProjectorMask(env, constants):

	def kvalues(dx, N):
		return numpy.fft.fftfreq(N, dx) * 2.0 * math.pi

	mask = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	# FIXME: use elementwise operations
	if constants.dim == 1:
		for k, kz in enumerate(kvalues(constants.dz, constants.nvz)):
			e_k = (constants.hbar * kz) ** 2 / (2.0 * constants.m)
			mask[k] = 0.0 if e_k > constants.e_cut else 1.0

	else:
		for i, kx in enumerate(kvalues(constants.dx, constants.nvx)):
			for j, ky in enumerate(kvalues(constants.dy, constants.nvy)):
				for k, kz in enumerate(kvalues(constants.dz, constants.nvz)):
					e_k = (constants.hbar ** 2) * \
						(kx * kx + ky * ky + kz * kz) / (2.0 * constants.m)

					mask[k, j, i] = 0.0 if e_k > constants.e_cut else 1.0

	modes = numpy.sum(mask)
	#print "Projector modes: " + str(modes) + " out of " + str(constants.cells)

	return env.toDevice(mask), modes
