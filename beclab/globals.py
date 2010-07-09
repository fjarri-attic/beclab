"""
Auxiliary functions and classes.
"""

try:
	import pyopencl as cl
	clBuffer = cl.Buffer
except:
	# when pyopencl is unavailable, buffer will not work anyway
	clBuffer = object

from mako.template import Template
import numpy

class _Buffer(clBuffer):
	"""
	Wrapper class for OpenCL buffer.
	Mimics some numpy array properties.
	"""

	def __init__(self, context, shape, dtype):
		self.size = 1
		for dim in shape:
			self.size *= dim

		if isinstance(dtype, numpy.dtype):
			self.itemsize = dtype.itemsize
		else:
			self.itemsize = dtype().nbytes

		self.nbytes = self.itemsize * self.size

		clBuffer.__init__(self, context, cl.mem_flags.READ_WRITE, size=self.nbytes)

		self.shape = shape
		self.dtype = dtype

	def reshape(self, new_shape):
		new_size = 1
		for dim in new_shape:
			new_size *= dim
		assert new_size == self.size
		self.shape = new_shape
		return self


class Environment:
	"""
	Abstraction layer for GPU/CPU context and memory management.
	"""

	def __init__(self, gpu):

		self.gpu = gpu

		if gpu:
			devices = []
			for platform in cl.get_platforms():
				devices.extend(platform.get_devices(device_type=cl.device_type.GPU))

			self.device = devices[0]
			self.context = cl.Context(devices=[self.device])
			self.queue = cl.CommandQueue(self.context)

	def allocate(self, shape, dtype):
		if self.gpu:
			return _Buffer(self.context, shape, dtype)
		else:
			return numpy.empty(shape, dtype=dtype)

	def synchronize(self):
		if self.gpu:
			self.queue.finish()

	def toCPU(self, buf, shape=None):
		if shape is None:
			shape = buf.shape

		if not self.gpu:
			return buf.reshape(shape)

		cpu_buf = numpy.empty(shape, dtype=buf.dtype)
		cl.enqueue_read_buffer(self.queue, buf, cpu_buf).wait()
		return cpu_buf

	def toGPU(self, buf, shape=None):
		if shape is None:
			shape = buf.shape

		if not self.gpu:
			return buf.reshape(shape)

		gpu_buf = _Buffer(self.context, shape, buf.dtype)
		cl.enqueue_write_buffer(self.queue, gpu_buf, buf).wait()
		return gpu_buf

	def copyBuffer(self, buf, dest=None):
		if self.gpu:
			if dest is None:
				buf_copy = self.allocate(buf.shape, buf.dtype)
			else:
				buf_copy = dest

			cl.enqueue_copy_buffer(self.queue, buf, buf_copy)

			if dest is None:
				return buf_copy
		else:
			if dest is None:
				return buf.copy()
			else:
				dest[:,:,:] = buf

	def __str__(self):
		if self.gpu:
			return "gpu"
		else:
			return "cpu"

	def compile(self, source, constants, **kwds):
		return _ProgramWrapper(self.context, self.queue, source, constants, **kwds)


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


class _ProgramWrapper:

	def __init__(self, context, queue, source, constants, **kwds):
		# program and kernels are tied to queue, which is not exactly logical,
		# but works for our purposes and makes code simpler (because program uses
		# single queue for all calculations anyway)
		self.queue = queue
		self._compile(context, source, constants, **kwds)

	def _compile(self, context, source, constants, **kwds):
		"""
		Adds helper functions and defines to given source, renders it,
		compiles and saves OpenCL program object.
		"""

		kernel_defines = Template("""
			${c.complex.name} complex_mul_scalar(${c.complex.name} a, ${c.scalar.name} b)
			{
				return ${c.complex.ctr}(a.x * b, a.y * b);
			}

			${c.complex.name} complex_mul(${c.complex.name} a, ${c.complex.name} b)
			{
				return ${c.complex.ctr}(mad(-a.y, b.y, a.x * b.x), mad(a.y, b.x, a.x * b.y));
			}

			${c.scalar.name} squared_abs(${c.complex.name} a)
			{
				return a.x * a.x + a.y * a.y;
			}

			${c.complex.name} conj(${c.complex.name} a)
			{
				return ${c.complex.ctr}(a.x, -a.y);
			}

			${c.complex.name} cexp(${c.complex.name} a)
			{
				${c.scalar.name} module = exp(a.x);
				${c.scalar.name} angle = a.y;
				return ${c.complex.ctr}(module * native_cos(angle), module * native_sin(angle));
			}

			float get_float_from_image(read_only image3d_t image, int i, int j, int k)
			{
				sampler_t sampler = CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP |
					CLK_NORMALIZED_COORDS_FALSE;

				uint4 image_data = read_imageui(image, sampler,
					(int4)(i, j, k, 0));

				return *((float*)&image_data);
			}

			#define DEFINE_INDEXES int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2), index = (k << ${c.nvx_pow + c.nvy_pow}) + (j << ${c.nvx_pow}) + i
		""")

		defines = kernel_defines.render(c=constants)
		kernel_src = Template(source).render(c=constants, **kwds)
		self._program = cl.Program(context, defines + kernel_src).build(options='-cl-mad-enable')

	def __getattr__(self, name):
		return _FunctionWrapper(getattr(self._program, name), self.queue)


class _FunctionWrapper:
	"""
	Wrapper for elementwise CL kernel. Caches prepared functions for
	calls with same element number.
	"""

	def __init__(self, kernel, queue):
		self._kernel = kernel
		self.queue = queue

	def __call__(self, shape, *args):
		self._kernel(self.queue, tuple(reversed(shape)), *args)


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
	"""Returns array with values of external potential energy."""

	potentials = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	for i in xrange(constants.nvx):
		for j in xrange(constants.nvy):
			for k in xrange(constants.nvz):
				x = -constants.xmax + i * constants.dx
				y = -constants.ymax + j * constants.dy
				z = -constants.zmax + k * constants.dz

				potentials[k, j, i] = (x * x + y * y + z * z /
					(constants.lambda_ * constants.lambda_)) / 2

	if not env.gpu:
		return potentials

	return cl.Image(env.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
		cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT32),
		shape=tuple(reversed(constants.shape)), hostbuf=potentials)

def getKVectors(env, constants):
	"""Returns array with values of k-space vectors."""

	def kvalue(i, dk, N):
		if 2 * i > N:
			return dk * (i - N)
		else:
			return dk * i

	kvectors = numpy.empty(constants.shape, dtype=constants.scalar.dtype)

	for i in xrange(constants.nvx):
		for j in xrange(constants.nvy):
			for k in xrange(constants.nvz):

				kx = kvalue(i, constants.dkx, constants.nvx)
				ky = kvalue(j, constants.dky, constants.nvy)
				kz = kvalue(k, constants.dkz, constants.nvz)

				kvectors[k, j, i] = (kx * kx + ky * ky + kz * kz) / 2

	if not env.gpu:
		return kvectors

	return cl.Image(env.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
		cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT32),
		shape=tuple(reversed(constants.shape)), hostbuf=kvectors)
