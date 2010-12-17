from pycuda.autoinit import device
from pycuda.driver import device_attribute
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.tools

import numpy
import os
from mako.template import Template

from .misc import log2

_dir, _file = os.path.split(os.path.abspath(__file__))
_header = Template(filename=os.path.join(_dir, 'header.mako'))


class _KernelWrapper:

	def __init__(self, env, kernel):
		self._env = env
		self._stream = env.stream
		self._kernel = kernel

		self._max_block_size = 2 ** log2(min(self._env.max_block_size,
			self._env.max_registers / self._kernel.num_regs))

	def __call__(self, size, *args):
		max_block_size = self._max_block_size
		block_size = min(size, max_block_size)

		if size <= block_size:
			block = (size, 1, 1)
			grid = (1, 1)
		else:
			block = (block_size, 1, 1)
			size /= block_size
			max_grid_size_x = self._env.max_grid_size_x
			if size > max_grid_size_x:
				grid = (max_grid_size_x, size / max_grid_size_x)
			else:
				grid = (size, 1)

		self._customCall(grid, block, *args)

	def _customCall(self, grid, block, *args):
		self._kernel.param_set(*args)
		self._kernel.set_block_shape(*block)
		self._kernel.launch_grid_async(grid[0], grid[1], self._stream)

	def customCall(self, global_size, block, *args):
		block = tuple(list(block) + [1] * (3 - len(block)))

		grid_x = global_size[0] / block[0]
		if len(global_size) == 1:
			grid = (grid_x, 1)
		else:
			grid = (grid_x, global_size[1] / block[1])

		self._customCall(grid, block, *args)


class _ProgramWrapper:

	def __init__(self, env, source, double=False, prelude="", **kwds):
		# program and kernels are tied to queue, which is not exactly logical,
		# but works for our purposes and makes code simpler (because program uses
		# single queue for all calculations anyway)
		self._env = env
		self._compile(source, double=double, prelude=prelude, **kwds)

	def _compile(self, source, double=False, prelude="", **kwds):
		"""
		Adds helper functions and defines to given source, renders it,
		compiles and saves OpenCL program object.
		"""
		kernel_src = Template(source).render(**kwds)
		src = _header.render(cuda=True, double=double, kernels=kernel_src, prelude=prelude)
		self._program = SourceModule(src, no_extern_c=True, options=['-use_fast_math'])

	def __getattr__(self, name):
		return _KernelWrapper(self._env, self._program.get_function(name))


class CUDAEnvironment:

	def __init__(self, device_num=0):

		cuda.init()

		#self.context = pycuda.tools.make_default_context()
		#self.device = self.context.get_device()

		self.device = cuda.Device(device_num)
		self.context = self.device.make_context()

		self.stream = cuda.Stream()

		self.max_block_size = self.device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X)

		self.max_grid_size_x = self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
		self.max_grid_size_y = self.device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)

		self.max_registers = self.device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_BLOCK)

		self.warp_size = self.device.get_attribute(cuda.device_attribute.WARP_SIZE)

		self.gpu = True
		self.cuda = True

	def allocate(self, shape, dtype):
		return gpuarray.GPUArray(shape, dtype=dtype)

	def synchronize(self):
		self.stream.synchronize()

	def fromDevice(self, buf, shape=None):
		res = buf.get()
		if shape is not None:
			res = res.reshape(shape)
		return res

	def toDevice(self, buf, shape=None, async=False):
		if shape is not None:
			buf = buf.reshape(shape)

		if async:
		# FIXME: there must be a warning in docs that buf has to be pagelocked
			return gpuarray.to_gpu_async(buf, stream=self.stream)
		else:
			return gpuarray.to_gpu(buf)

	def copyBuffer(self, buf, dest=None):
		if dest is None:
			buf_copy = self.allocate(buf.shape, buf.dtype)
		else:
			buf_copy = dest

		cuda.memcpy_dtod_async(buf_copy.gpudata, buf.gpudata, buf.nbytes, stream=self.stream)

		if dest is None:
			return buf_copy

	def __str__(self):
		return "CUDA"

	def release(self):
		self.context.pop()

	def compile(self, source, double=False, prelude="", **kwds):
		return _ProgramWrapper(self, source, double=double, prelude=prelude, **kwds)
