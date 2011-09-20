import numpy


class Parameters(dict):

	def __init__(self, **kwds):
		dict.__init__(self, **kwds)
		self._default_keys = set(kwds.keys())

	def add_defaults(self, kwds):
		self._default_keys.update(kwds.keys())
		self.update(kwds)

	def need_update(self, other):
		for key in other:
			assert key in self._default_keys, "Unknown key: " + key
			if self[key] != other[key]:
				return True

		return False

	def safe_update(self, kwds):
		for key in kwds:
			assert key in self._default_keys, "Unknown key: " + key
		self.update(kwds)

	def __getattr__(self, attr):
		return self[attr]

	def __setattr__(self, attr, value):
		self[attr] = value


class PairedCalculation(object):
	"""
	Base class for paired GPU/CPU calculations.
	Depending on initializing parameter, it will make visible either _gpu_
	or _cpu_ methods.
	"""

	def __init__(self, env):
		self.__gpu = env.gpu
		self.__createAliases()
		self._env = env
		self._p = None
		self.__prepared = False

	def _addParameters(self, *args, **kwds):
		if self._p is None:
			self._p = Parameters(**kwds)
		else:
			self._p.add_defaults(kwds)

		if len(args) > 0:
			self._p.safe_update(args[0])

	def _prepare(self):
		pass

	def _prepare_specific(self):
		pass

	def _prepare_hierarchy(self):
		super(parent_cls)._prepare_hierarchy()

	def prepare(self, **kwds):
		if self.__prepared and not self._p.need_update(kwds):
			return

		self.__prepared = True
		self._p.safe_update(kwds)

		for c in reversed(type(self).__mro__):
			if hasattr(c, '_prepare'):
				c._prepare(self)
				if self._env.gpu:
					if hasattr(c, '_gpu__prepare_specific'): c._gpu__prepare_specific(self)
				else:
					if hasattr(c, '_cpu__prepare_specific'): c._cpu__prepare_specific(self)

		return self

	def compileProgram(self, template, **kwds):
		return self._env.compileProgram(template, self._constants, self._grid, p=self._p, **kwds)

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

def tile2D(x, y):
	nx = len(x)
	ny = len(y)

	xx = numpy.tile(x, ny).reshape(ny, nx)
	yy = numpy.tile(y, nx).reshape(nx, ny).transpose()

	return xx, yy

def reshape(arr, *shape):
	# TODO: add more error-checking, perhaps
	if isinstance(shape[0], tuple) or isinstance(shape[0], list):
		shape = tuple(shape[0])
	size = reduce(lambda x, y: x * y, shape, 1)
	if size != arr.size:
	   raise ValueError("total size of new array must be unchanged")

	from pycuda.gpuarray import GPUArray
	return GPUArray(
			shape=shape,
			dtype=arr.dtype,
			allocator=arr.allocator,
			base=arr,
			gpudata=int(arr.gpudata))

def ravel(arr):
	return reshape(arr, arr.size)

def view(arr, dtype=None):
	if dtype is None:
		dtype = arr.dtype

	old_itemsize = arr.dtype.itemsize
	itemsize = numpy.dtype(dtype).itemsize

	if arr.shape[-1] * old_itemsize % itemsize != 0:
		raise ValueError("new type not compatible with array")

	shape = arr.shape[:-1] + (arr.shape[-1] * old_itemsize // itemsize,)

	from pycuda.gpuarray import GPUArray
	return GPUArray(
			shape=shape,
			dtype=dtype,
			allocator=arr.allocator,
			base=arr,
			gpudata=int(arr.gpudata))

def getView(arr, shape, offset=0, dtype=None):
	new_size = reduce(lambda x, y: x * y, shape)

	# workaround for numpy bug
	if dtype is None:
		dtype = arr.dtype

	# FIXME: temporary solution to avoid unnecessary import of pycuda
	if isinstance(arr, numpy.ndarray):
		return arr.ravel()[offset:].view(dtype)[:new_size].reshape(shape)
	else:
		return reshape(view(ravel(arr)[offset:], dtype)[:new_size], shape)

def elementwiseMatrixExp(a):
	assert a.shape[:2] == (2, 2)

	delta = numpy.sqrt((a[0, 0] - a[1, 1]) ** 2 + 4 * a[0, 1] * a[1, 0])
	sinh_d = numpy.sinh(delta / 2)
	cosh_d = numpy.cosh(delta / 2)
	t = numpy.exp((a[0, 0] + a[1, 1]) / 2)

	m11 = t * (delta * cosh_d + (a[0, 0] - a[1, 1]) * sinh_d)
	m12 = 2 * a[0, 1] * t * sinh_d
	m21 = 2 * a[1, 0] * t * sinh_d
	m22 = t * (delta * cosh_d + (a[1, 1] - a[0, 0]) * sinh_d)

	res = numpy.array([[m11, m12], [m21, m22]])

	# Some elements of delta may equal to 0; in this case
	# we have to process these positions taking the limit of delta -> 0
	res_limit = t * numpy.array([
		[1 + (a[0, 0] - a[1, 1]) / 2, a[0, 1]],
		[a[1, 0], 1 - (a[0, 0] - a[1, 1]) / 2]
	])

	func = lambda r, r_limit, d: r / d if d != 0 else r_limit
	vfunc = numpy.vectorize(func)

	return vfunc(res, res_limit, delta)
