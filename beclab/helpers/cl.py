import pyopencl as cl

import numpy
import os
from mako.template import Template

_dir, _file = os.path.split(os.path.abspath(__file__))
_header = Template(filename=os.path.join(_dir, 'header.mako'))


class _Buffer(cl.Buffer):
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

		cl.Buffer.__init__(self, context, cl.mem_flags.READ_WRITE, size=self.nbytes)

		self.shape = shape
		self.dtype = dtype

	def reshape(self, new_shape):
		new_size = 1
		for dim in new_shape:
			new_size *= dim
		assert new_size == self.size
		self.shape = new_shape
		return self


class _KernelWrapper:

	def __init__(self, kernel, queue):
		self._kernel = kernel
		self._queue = queue

	def __call__(self, size, *args):
		self._kernel(self._queue, (size,), None, *args)

	def customCall(self, global_size, block, *args):
		self._kernel(self._queue, global_size, block, *args)



class _ProgramWrapper:

	def __init__(self, context, queue, source, double=False, prelude="", **kwds):
		# program and kernels are tied to queue, which is not exactly logical,
		# but works for our purposes and makes code simpler (because program uses
		# single queue for all calculations anyway)
		self.queue = queue
		self._compile(context, source, double=double, prelude=prelude, **kwds)

	def _compile(self, context, source, double=False, prelude="", **kwds):
		"""
		Adds helper functions and defines to given source, renders it,
		compiles and saves OpenCL program object.
		"""
		kernel_src = Template(source).render(**kwds)
		src = _header.render(cuda=False, double=double, kernels=kernel_src, prelude=prelude)
		self._program = cl.Program(context, src).build(options='-cl-mad-enable')

	def __getattr__(self, name):
		return _KernelWrapper(getattr(self._program, name), self.queue)


class CLEnvironment:

	def __init__(self):
		devices = []
		for platform in cl.get_platforms():
			devices.extend(platform.get_devices(device_type=cl.device_type.GPU))

		self.device = devices[0]
		self.context = cl.Context(devices=[self.device])
		self.queue = cl.CommandQueue(self.context)

		self.max_block_size = self.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0]
		self.warp_size = 32 # FIXME: must get it from device

		self.gpu = True
		self.cuda = False

	def allocate(self, shape, dtype):
		return _Buffer(self.context, shape, dtype)

	def synchronize(self):
		self.queue.finish()

	def fromDevice(self, buf, shape=None):
		if shape is not None:
			shape = buf.shape

		cpu_buf = numpy.empty(shape, dtype=buf.dtype)
		cl.enqueue_read_buffer(self.queue, buf, cpu_buf).wait()
		return cpu_buf

	def toDevice(self, buf, shape=None, async=False):
		if shape is not None:
			shape = buf.shape

		gpu_buf = _Buffer(self.context, shape, buf.dtype)
		event = cl.enqueue_write_buffer(self.queue, gpu_buf, buf)
		if not async:
			event.wait()
		return gpu_buf

	def copyBuffer(self, buf, dest=None):
		if dest is None:
			buf_copy = self.allocate(buf.shape, buf.dtype)
		else:
			buf_copy = dest

		cl.enqueue_copy_buffer(self.queue, buf, buf_copy)

		if dest is None:
			return buf_copy

	def __str__(self):
		return "CL"

	def release(self):
		pass

	def compile(self, source, double=False, prelude="", **kwds):
		return _ProgramWrapper(self.context, self.queue, source, double=double,
			prelude=prelude, **kwds)
