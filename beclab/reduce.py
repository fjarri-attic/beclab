try:
	import pyopencl as cl
except:
	pass

from mako.template import Template
import numpy

from .globals import *
from .transpose import Transpose

class Reduce:

	def __init__(self, env, constants):
		self._env = env
		self._constants = constants
		self._tr_scalar = Transpose(env, constants.scalar)
		self._tr_complex = Transpose(env, constants.complex)

		kernel_template = Template("""
		%for typename in (c.scalar.name, c.complex.name):
		%for block_size in [2 ** x for x in xrange(log2(max_block_size) + 1)]:
			<%
				log2_warp_size = log2(warp_size)
				log2_block_size = log2(block_size)
				if block_size > warp_size:
					smem_size = block_size
				else:
					smem_size = block_size + block_size / 2
			%>

			__kernel void reduceKernel${block_size}${typename}(
				__global ${typename}* output, const __global ${typename}* input)
			{
				__local ${typename} shared_mem[${smem_size}];

				int tid = get_local_id(0);
				int bid = get_group_id(0);

				// first reduction, after which the number of elements to reduce
				// equals to number of threads in block
				shared_mem[tid] = input[tid + 2 * bid * ${block_size}] +
					input[tid + 2 * bid * ${block_size} + ${block_size}];

				barrier(CLK_LOCAL_MEM_FENCE);

				// 'if(tid)'s will split execution only near the border of warps,
				// so they are not affecting performance (i.e, for each warp there
				// will be only one path of execution anyway)
				%for reduction_pow in xrange(log2_block_size - 1, log2_warp_size, -1):
					if(tid < ${2 ** reduction_pow})
						shared_mem[tid] += shared_mem[tid + ${2 ** reduction_pow}];
					barrier(CLK_LOCAL_MEM_FENCE);
				%endfor

				// The following code will be executed inside a single warp, so no
				// shared memory synchronization is necessary
				if (tid < ${warp_size}) {
				%for reduction_pow in xrange(log2_warp_size, -1, -1):
					%if block_size >= 2 ** (reduction_pow + 1):
						shared_mem[tid] += shared_mem[tid + ${2 ** reduction_pow}];
					%endif
				%endfor
				}

				if (tid == 0) output[bid] = shared_mem[0];
			}
		%endfor

		%for reduce_power in [2 ** x for x in xrange(1, log2(warp_size / 2))]:
		__kernel void smallSparseReduce${reduce_power}${typename}(__global ${typename}* output, const __global ${typename}* input)
		{
			int id = get_global_id(0);
			int curr_id = id;
			int size = get_global_size(0);
			${typename} temp = input[curr_id];

			%for i in xrange(1, reduce_power):
				curr_id += size;
				temp += input[curr_id];
			%endfor

			output[id] = temp;
		}
		%endfor

		%endfor
		""")

		workgroup_sizes = self._env.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
		self._max_block_size = workgroup_sizes[0]

		self._warp_size = 32

		kernel_src = kernel_template.render(c=self._constants,
			warp_size=self._warp_size, max_block_size=self._max_block_size, log2=log2)
		program = cl.Program(self._env.context, kernel_src).build()

		self._scalar_kernels = {}
		for local_size in [2 ** x for x in xrange(log2(self._max_block_size) + 1)]:
			name = "reduceKernel" + str(local_size) + self._constants.scalar.name
			self._scalar_kernels[local_size] = getattr(program, name)

		self._complex_kernels = {}
		for local_size in [2 ** x for x in xrange(log2(self._max_block_size) + 1)]:
			name = "reduceKernel" + str(local_size) + self._constants.complex.name
			self._complex_kernels[local_size] = getattr(program, name)

		self._small_scalar_kernels = {}
		for reduce_power in [2 ** x for x in xrange(1, log2(self._warp_size / 2))]:
			name = "smallSparseReduce" + str(reduce_power) + self._constants.scalar.name
			self._small_scalar_kernels[reduce_power] = getattr(program, name)

		self._small_complex_kernels = {}
		for reduce_power in [2 ** x for x in xrange(1, log2(self._warp_size / 2))]:
			name = "smallSparseReduce" + str(reduce_power) + self._constants.complex.name
			self._small_complex_kernels[reduce_power] = getattr(program, name)

	def __call__(self, array, final_length=1):

		length = array.size
		assert length >= final_length, "Array size cannot be less than final size"

		if array.dtype == self._constants.scalar.dtype:
			reduce_kernels = self._scalar_kernels
			itemsize = self._constants.scalar.nbytes
		else:
			reduce_kernels = self._complex_kernels
			itemsize = self._constants.complex.nbytes

		if length == final_length:
			res = self._env.allocate((length,), array.dtype)
			cl.enqueue_copy_buffer(self._env.queue, array, res, length * itemsize)
			return res

		# we can reduce maximum block size * 2 times a pass
		max_reduce_power = self._max_block_size * 2

		data_in = array

		while length > final_length:

			if length / final_length >= max_reduce_power:
				reduce_power = max_reduce_power
			else:
				reduce_power = length / final_length

			data_out = self._env.allocate((data_in.size / reduce_power,), array.dtype)

			func = reduce_kernels[reduce_power / 2]

			func(self._env.queue, (length / 2,), data_out, data_in, local_size=(reduce_power / 2,))

			length /= reduce_power

			data_in = data_out

		if final_length == 1:
		# return reduction result
			result = numpy.array((1,), dtype=array.dtype)
			cl.enqueue_read_buffer(self._env.queue, data_in, result)
			self._env.queue.finish()
			return result[0]
		else:
			return data_in

	def sparse(self, array, final_length=1):
		if final_length == 1:
			return self(array)

		reduce_power = array.size / final_length

		if reduce_power == 1:
			res = self._env.allocate((final_length,), array.dtype)
			cl.enqueue_copy_buffer(self._env.queue, array, res)
			return res
		if reduce_power < self._warp_size / 2:
			res = self._env.allocate((final_length,), array.dtype)
			if array.dtype == self._constants.scalar.dtype:
				func = self._small_scalar_kernels[reduce_power]
			else:
				func = self._small_complex_kernels[reduce_power]
			func(self._env.queue, (final_length,), res, array)
			return res
		else:
			res = self._env.allocate(array.shape, array.dtype)
			if array.dtype == self._constants.scalar.dtype:
				self._tr_scalar(res, array, final_length, reduce_power)
			else:
				self._tr_complex(res, array, final_length, reduce_power)

			return self(res, final_length=final_length)


class CPUReduce:

	def __call__(self, array, final_length=1):

		if final_length == 1:
			return numpy.sum(array)

		flat_array = array.ravel()
		res = numpy.empty(final_length)
		reduce_power = array.size / final_length

		for i in xrange(final_length):
			res[i] = numpy.sum(flat_array[i*reduce_power:(i+1)*reduce_power])

		return res

	def sparse(self, array, final_length=1):

		if final_length == 1:
			return self(array)

		reduce_power = array.size / final_length
		return self(numpy.transpose(array.reshape(reduce_power, final_length)), final_length=final_length)


def getReduce(env, constants):
	if env.gpu:
		return Reduce(env, constants)
	else:
		return CPUReduce()
