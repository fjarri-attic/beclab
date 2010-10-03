import numpy

from .misc import log2
from .transpose import createTranspose
from .typenames import MAP

class GPUReduce:

	def __init__(self, env, dtype):
		self._env = env
		self._tr = createTranspose(env, dtype)

		type = MAP[dtype]

		kernel_template = """
		%for block_size in block_sizes:
			<%
				log2_warp_size = log2(warp_size)
				log2_block_size = log2(block_size)
				if block_size > warp_size:
					smem_size = block_size
				else:
					smem_size = block_size + block_size / 2
			%>

			EXPORTED_FUNC void reduceKernel${block_size}(
				GLOBAL_MEM ${typename}* output, const GLOBAL_MEM ${typename}* input)
			{
				SHARED_MEM ${typename} shared_mem[${smem_size}];

				int tid = THREAD_ID_X;
				int bid = BLOCK_ID_X;

				// first reduction, after which the number of elements to reduce
				// equals to number of threads in block
				shared_mem[tid] = input[tid + 2 * bid * ${block_size}] +
					input[tid + 2 * bid * ${block_size} + ${block_size}];

				SYNC;

				// 'if(tid)'s will split execution only near the border of warps,
				// so they are not affecting performance (i.e, for each warp there
				// will be only one path of execution anyway)
				%for reduction_pow in xrange(log2_block_size - 1, log2_warp_size, -1):
					if(tid < ${2 ** reduction_pow})
						shared_mem[tid] = shared_mem[tid] + shared_mem[tid + ${2 ** reduction_pow}];
					SYNC;
				%endfor

				// The following code will be executed inside a single warp, so no
				// shared memory synchronization is necessary
				if (tid < ${warp_size}) {
				%for reduction_pow in xrange(log2_warp_size, -1, -1):
					%if block_size >= 2 ** (reduction_pow + 1):
						shared_mem[tid] = shared_mem[tid] + shared_mem[tid + ${2 ** reduction_pow}];
					%endif
				%endfor
				}

				if (tid == 0) output[bid] = shared_mem[0];
			}
		%endfor

		%for reduce_power in reduce_powers:
		EXPORTED_FUNC void smallSparseReduce${reduce_power}(
			GLOBAL_MEM ${typename}* output, const GLOBAL_MEM ${typename}* input)
		{
			int id = GLOBAL_ID_X;
			int curr_id = id;
			int size = GLOBAL_SIZE_X;
			${typename} temp = input[curr_id];

			%for i in xrange(1, reduce_power):
				curr_id += size;
				temp = temp + input[curr_id];
			%endfor

			output[id] = temp;
		}
		%endfor
		"""

		self._max_block_size = self._env.max_block_size
		self._warp_size = self._env.warp_size

		block_sizes = [2 ** x for x in xrange(log2(self._max_block_size) + 1)]
		reduce_powers = [2 ** x for x in xrange(1, log2(self._warp_size / 2))]

		program = self._env.compile(kernel_template, double=type.precision.double,
			typename=type.name, warp_size=self._warp_size,
			max_block_size=self._max_block_size,
			log2=log2, block_sizes=block_sizes, reduce_powers=reduce_powers)

		self._kernels = {}
		for block_size in block_sizes:
			name = "reduceKernel" + str(block_size)
			self._kernels[block_size] = getattr(program, name)

		self._small_kernels = {}
		for reduce_power in reduce_powers:
			name = "smallSparseReduce" + str(reduce_power)
			self._small_kernels[reduce_power] = getattr(program, name)

	def __call__(self, array, final_length=1):

		length = array.size
		assert length >= final_length, "Array size cannot be less than final size"

		reduce_kernels = self._kernels

		if length == final_length:
			res = self._env.allocate((length,), array.dtype)
			self._env.copyBuffer(array, res)
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

			func.customCall((length / 2,), (reduce_power / 2,), data_out, data_in)

			length /= reduce_power

			data_in = data_out

		if final_length == 1:
		# return reduction result
			return self._env.fromDevice(data_in)[0]
		else:
			return data_in

	def sparse(self, array, final_length=1):
		if final_length == 1:
			return self(array)

		reduce_power = array.size / final_length

		if reduce_power == 1:
			res = self._env.allocate((final_length,), array.dtype)
			self._env.copyBuffer(array, res)
			return res
		if reduce_power < self._warp_size / 2:
			res = self._env.allocate((final_length,), array.dtype)
			func = self._small_kernels[reduce_power]
			func(final_length, res, array)
			return res
		else:
			res = self._env.allocate(array.shape, array.dtype)
			self._tr(array, res, final_length, reduce_power)
			return self(res, final_length=final_length)


class CPUReduce:

	def __call__(self, array, final_length=1):

		if final_length == 1:
			return numpy.sum(array)

		flat_array = array.ravel()
		res = numpy.empty(final_length, dtype=array.dtype)
		reduce_power = array.size / final_length

		for i in xrange(final_length):
			res[i] = numpy.sum(flat_array[i*reduce_power:(i+1)*reduce_power])

		return res

	def sparse(self, array, final_length=1):

		if final_length == 1:
			return self(array)

		reduce_power = array.size / final_length
		return self(array.reshape(reduce_power, final_length).T, final_length=final_length)


def createReduce(env, dtype):
	if env.gpu:
		return GPUReduce(env, dtype)
	else:
		return CPUReduce()
