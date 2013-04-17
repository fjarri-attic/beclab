from pycuda.autoinit import device
from pycuda.driver import device_attribute
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.tools

import numpy
import os
import gc
from mako.template import Template

from .misc import log2

_dir, _file = os.path.split(os.path.abspath(__file__))
_header = Template(filename=os.path.join(_dir, 'header.mako'))


class _KernelWrapper:

	def __init__(self, env, kernel, sync_calls):
		self._env = env
		self._stream = env.stream
		self._kernel = kernel
		self._sync_calls = sync_calls

		self.max_threads_per_block = kernel.max_threads_per_block

		if self._kernel.num_regs > 0:
			self._max_block_size = 2 ** log2(min(self._env.max_block_size,
				self._env.max_registers / self._kernel.num_regs))
		else:
			self._max_block_size = self._env.max_block_size

	def _getSupportedGrid(self, x, y):
		max_x = self._env.max_grid_size_x_pow2
		if x > max_x:
			return (max_x, (y * x - 1) / max_x + 1)
		else:
			return (x, y)

	def __call__(self, size, *args):
		max_block_size = self._max_block_size
		block_size = min(size, max_block_size)

		if size <= block_size:
			block = (size, 1, 1)
			grid = (1, 1)
		else:
			block = (block_size, 1, 1)
			grid = self._getSupportedGrid((size - 1) / block_size + 1, 1)

		self._customCall(grid, block, numpy.int32(size), *args)

	def _customCall(self, grid, block, *args):
		self._kernel(*args, grid=grid, block=block, stream=self._stream)
		if self._sync_calls: self._env.synchronize()

	def customCall(self, global_size, block, *args):
		block = tuple(list(block) + [1] * (3 - len(block)))

		grid_x = global_size[0] / block[0]
		if len(global_size) == 1:
			grid = (grid_x, 1)
		else:
			grid = (grid_x, global_size[1] / block[1])

		grid = self._getSupportedGrid(*grid)

		self._customCall(grid, block, *args)


class _ProgramWrapper:

	def __init__(self, env, source, sync_calls, double=False, prelude="", **kwds):
		# program and kernels are tied to queue, which is not exactly logical,
		# but works for our purposes and makes code simpler (because program uses
		# single queue for all calculations anyway)
		self._env = env
		self._compile(source, double=double, prelude=prelude, **kwds)
		self._sync_calls = sync_calls

	def _compile(self, source, double=False, prelude="", manual_extern_c=False, **kwds):
		"""
		Adds helper functions and defines to given source, renders it,
		compiles and saves OpenCL program object.
		"""
		kernel_src = Template(source).render(**kwds)
		src = _header.render(cuda=True, double=double, kernels=kernel_src, prelude=prelude,
			manual_extern_c=manual_extern_c)
		try:
			self._program = SourceModule(src, no_extern_c=True, options=['-use_fast_math'])
		except:
			for i, l in enumerate(src.split('\n')):
				print i + 1, ": ", l
			raise

	def __getattr__(self, name):
		return _KernelWrapper(self._env, self._program.get_function(name), self._sync_calls)


class CUDAEnvironment:

	def __init__(self, device_num=0, sync_calls=False):

		cuda.init()

		#self.context = pycuda.tools.make_default_context()
		#self.device = self.context.get_device()

		self.device = cuda.Device(device_num)
		self.context = self.device.make_context()

		self.stream = cuda.Stream()

		self.max_block_size = self.device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X)

		self.max_grid_size_x = self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
		self.max_grid_size_y = self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)

		self.max_grid_size_x_pow2 = 2 ** log2(self.max_grid_size_x)

		self.max_registers = self.device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_BLOCK)

		self.warp_size = self.device.get_attribute(cuda.device_attribute.WARP_SIZE)

		self.gpu = True
		self.cuda = True

		self._sync_calls = sync_calls

		self.allocated = 0

	def allocate(self, shape, dtype):

		size = 0
		try:
			size += dtype().itemsize
		except:
			size += dtype.itemsize

		for n in shape:
			size *= n

		self.allocated += size
#		print "Allocating", size

		return gpuarray.GPUArray(shape, dtype=dtype)

	def synchronize(self):
		self.stream.synchronize()

	def fromDevice(self, buf, shape=None):
		res = buf.get()
		if shape is not None:
			res = res.reshape(shape)
		return res

	def toDevice(self, buf, shape=None, async=False, dest=None):
		if shape is not None:
			buf = buf.reshape(shape)

		if dest is None:
			if async:
			# FIXME: there must be a warning in docs that buf has to be pagelocked
				return gpuarray.to_gpu_async(buf, stream=self.stream)
			else:
				return gpuarray.to_gpu(buf)
		else:
			cuda.memcpy_htod_async(dest.gpudata,
				buf, stream=None)

	def copyBuffer(self, buf, dest, src_offset=0, dest_offset=0, length=None):

		elem_size = buf.dtype.itemsize
		size = buf.nbytes if length is None else elem_size * length
		src_offset *= elem_size
		dest_offset *= elem_size

		cuda.memcpy_dtod_async(int(dest.gpudata) + dest_offset,
			int(buf.gpudata) + src_offset,
			size, stream=self.stream)

		if dest is None:
			return buf_copy

	def __str__(self):
		return "CUDA"

	def release(self):
#		print "Total allocated:", self.allocated
		self.context.pop()
		gc.collect() # forcefully frees all buffers on GPU

	def compile(self, source, double=False, prelude="", **kwds):
		return _ProgramWrapper(self, source, self._sync_calls,
			double=double, prelude=prelude, **kwds)

	def supportsDouble(self):
		major, minor = self.context.get_device().compute_capability()
		return (major == 1 and minor == 3) or major >= 2
