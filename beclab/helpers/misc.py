import numpy

class PairedCalculation:
	"""
	Base class for paired GPU/CPU calculations.
	Depending on initializing parameter, it will make visible either _gpu_
	or _cpu_ methods.
	"""

	def __init__(self, env):
		self.__gpu = env.gpu
		self.__createAliases()
		self._env = env

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
