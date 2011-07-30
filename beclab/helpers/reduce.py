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
				GLOBAL_MEM ${typename}* output, const GLOBAL_MEM ${typename}* input,
				int blocks_per_part, int last_block_size)
			{
				SHARED_MEM ${typename} shared_mem[${smem_size}];

				int tid = THREAD_ID_X;
				int bid = BLOCK_ID_FLAT;

				int part_length = (blocks_per_part - 1) * blockDim.x + last_block_size;
				int part_num = BLOCK_ID_FLAT / blocks_per_part;
				int index_in_part = blockDim.x * (BLOCK_ID_FLAT % blocks_per_part) + tid;

				if(bid % blocks_per_part == blocks_per_part - 1 && tid >= last_block_size)
					shared_mem[tid] = ${construct_zero};
				else
					shared_mem[tid] = input[part_length * part_num + index_in_part];

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
				%if log2_block_size > 0:
				if (tid < ${warp_size}) {
				#ifdef CUDA
				// Fix for Fermi videocards, see Compatibility Guide 1.2.2
				volatile ${typename} *smem = shared_mem;
				#else
				SHARED_MEM ${typename} *smem = shared_mem;
				#endif
				%for reduction_pow in xrange(min(log2_warp_size, log2_block_size - 1), -1, -1):
					## nvcc considers 'volatile float2' and 'float2' to be incompatible
					## works well with scalars though
					%if typename.endswith('2'):
						smem[tid].x += smem[tid + ${2 ** reduction_pow}].x;
						smem[tid].y += smem[tid + ${2 ** reduction_pow}].y;
					%else:
						smem[tid] += smem[tid + ${2 ** reduction_pow}];
					%endif
				%endfor
				}
				%endif

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
			log2=log2, block_sizes=block_sizes, reduce_powers=reduce_powers,
			construct_zero="0" if type.dtype in (numpy.float32, numpy.float64) else "complex_ctr(0, 0)")

		self._kernels = {}
		for block_size in block_sizes:
			name = "reduceKernel" + str(block_size)
			self._kernels[block_size] = getattr(program, name)

		self._small_kernels = {}
		for reduce_power in reduce_powers:
			name = "smallSparseReduce" + str(reduce_power)
			self._small_kernels[reduce_power] = getattr(program, name)

	def __call__(self, array, length=None, final_length=1):

		if length is None:
			length = array.size

		assert length >= final_length, "Array size cannot be less than final size"
		assert length % final_length == 0

		reduce_kernels = self._kernels

		if length == final_length:
			res = self._env.allocate((length,), array.dtype)
			self._env.copyBuffer(array, res)
			return res

		# we can reduce maximum 'block size' times a pass
		max_reduce_power = self._max_block_size

		data_in = array

		while length > final_length:

			part_length = length / final_length

			if length / final_length >= max_reduce_power:
				block_size = max_reduce_power
				blocks_per_part = (part_length - 1) / block_size + 1
				blocks_num = blocks_per_part * final_length
				last_block_size = part_length - (blocks_per_part - 1) * block_size
				new_length = blocks_num
			else:
				block_size = 2 ** (log2(length / final_length - 1) + 1)
				blocks_per_part = 1
				blocks_num = final_length
				last_block_size = length / final_length
				new_length = final_length

			#print length, part_length, block_size, blocks_per_part, blocks_num, last_block_size

			grid_size = blocks_num * block_size
			data_out = self._env.allocate((new_length,), array.dtype)

			func = reduce_kernels[block_size]

			func.customCall((grid_size,), (block_size,), data_out, data_in,
				numpy.int32(blocks_per_part), numpy.int32(last_block_size))

			length = new_length

			data_in = data_out

#		if final_length == 1:
		# return reduction result
#			return self._env.fromDevice(data_in)[0]
#		else:
		return data_in

	def sparse(self, array, final_length=1, final_shape=None):
		if final_length == 1:
			return self(array)

		reduce_power = array.size / final_length

		# TODO: check that final_shape contains necessary amount of elements
		if final_shape is None:
			final_shape = (final_length,)

		if reduce_power == 1:
			res = self._env.allocate(final_shape, array.dtype)
			self._env.copyBuffer(array, res)
			return res
		# TODO: we can improve performance for small sizes by using special kernel
		#if reduce_power < self._warp_size / 2:
		#	res = self._env.allocate(final_shape, array.dtype)
		#	func = self._small_kernels[reduce_power]
		#	func(final_length, res, array)
		#	return res
		else:
			res = self._env.allocate(array.shape, array.dtype)
			self._tr(array, res, final_length, reduce_power)
			return self(res, final_length=final_length)


class GPUMaxFinder:

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

			// maximum
			#define MAX(a, b) ((a) < (b) ? (b) : (a))
			%if typename.endswith('2'):
			#define OP(a, b) complex_ctr(MAX(a.x, b.x), MAX(a.y, b.y))
			%else:
			#define OP(a, b) MAX((a), (b))
			%endif

			EXPORTED_FUNC void reduceKernel${block_size}(
				GLOBAL_MEM ${typename}* output, const GLOBAL_MEM ${typename}* input,
				int blocks_per_part, int last_block_size)
			{
				SHARED_MEM ${typename} shared_mem[${smem_size}];

				int tid = THREAD_ID_X;
				int bid = BLOCK_ID_FLAT;

				int part_length = (blocks_per_part - 1) * blockDim.x + last_block_size;
				int part_num = BLOCK_ID_FLAT / blocks_per_part;
				int index_in_part = blockDim.x * (BLOCK_ID_FLAT % blocks_per_part) + tid;

				if(bid % blocks_per_part == blocks_per_part - 1 && tid >= last_block_size)
					shared_mem[tid] = ${construct_zero};
				else
					shared_mem[tid] = input[part_length * part_num + index_in_part];

				SYNC;

				// 'if(tid)'s will split execution only near the border of warps,
				// so they are not affecting performance (i.e, for each warp there
				// will be only one path of execution anyway)
				%for reduction_pow in xrange(log2_block_size - 1, log2_warp_size, -1):
					if(tid < ${2 ** reduction_pow})
						shared_mem[tid] = OP(shared_mem[tid],
							shared_mem[tid + ${2 ** reduction_pow}]);
					SYNC;
				%endfor

				// The following code will be executed inside a single warp, so no
				// shared memory synchronization is necessary
				%if log2_block_size > 0:
				if (tid < ${warp_size}) {
				#ifdef CUDA
				// Fix for Fermi videocards, see Compatibility Guide 1.2.2
				volatile ${typename} *smem = shared_mem;
				#else
				SHARED_MEM ${typename} *smem = shared_mem;
				#endif

				${typename} ttt;
				%for reduction_pow in xrange(min(log2_warp_size, log2_block_size - 1), -1, -1):
				ttt = OP(smem[tid], smem[tid + ${2 ** reduction_pow}]);
				%if typename.endswith('2'):
					smem[tid].x = ttt.x;
					smem[tid].y = ttt.y;
				%else:
					smem[tid] = ttt;
				%endif
				%endfor
				}
				%endif

				if (tid == 0) output[bid] = shared_mem[0];
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
			log2=log2, block_sizes=block_sizes, reduce_powers=reduce_powers,
			construct_zero="0" if type.dtype in (numpy.float32, numpy.float64) else "complex_ctr(0, 0)")

		self._kernels = {}
		for block_size in block_sizes:
			name = "reduceKernel" + str(block_size)
			self._kernels[block_size] = getattr(program, name)

	def __call__(self, array, length=None):

		if length is None:
			length = array.size
		final_length = 1

		assert length >= final_length, "Array size cannot be less than final size"
		assert length % final_length == 0

		reduce_kernels = self._kernels

		if length == final_length:
			res = self._env.allocate((length,), array.dtype)
			self._env.copyBuffer(array, res)
			return res

		# we can reduce maximum 'block size' times a pass
		max_reduce_power = self._max_block_size

		data_in = array

		while length > final_length:

			part_length = length / final_length

			if length / final_length >= max_reduce_power:
				block_size = max_reduce_power
				blocks_per_part = (part_length - 1) / block_size + 1
				blocks_num = blocks_per_part * final_length
				last_block_size = part_length - (blocks_per_part - 1) * block_size
				new_length = blocks_num
			else:
				block_size = 2 ** (log2(length / final_length - 1) + 1)
				blocks_per_part = 1
				blocks_num = final_length
				last_block_size = length / final_length
				new_length = final_length

			#print length, part_length, block_size, blocks_per_part, blocks_num, last_block_size

			grid_size = blocks_num * block_size
			data_out = self._env.allocate((new_length,), array.dtype)

			func = reduce_kernels[block_size]

			func.customCall((grid_size,), (block_size,), data_out, data_in,
				numpy.int32(blocks_per_part), numpy.int32(last_block_size))

			length = new_length

			data_in = data_out

		if final_length == 1:
		# return reduction result
			x = self._env.fromDevice(data_in)[0]
			return max(x.real, x.imag)
		else:
			return data_in


class CPUReduce:

	def __call__(self, array, length=None, final_length=1):

		if final_length == 1:
			return numpy.sum(array.ravel()[:length]).reshape((1,))

		if length is None:
			length = array.size

		return array.ravel()[:length].reshape(final_length, length / final_length).sum(1)

	def sparse(self, array, length=None, final_length=1, final_shape=None):

		if final_length == 1:
			return self(array)

		if length is None:
			length = array.size

		# TODO: check that final_shape contains necessary amount of elements
		if final_shape is None:
			final_shape = (final_length,)

		reduce_power = length / final_length
		return self(array.ravel()[:length].reshape(reduce_power, final_length).T,
			final_length=final_length).reshape(final_shape)


class CPUMaxFinder:

	def __call__(self, array, length=None):

		a = array.ravel()[:length] if length is not None else array
		return max(a.real.max(), a.imag.max())


def createReduce(env, dtype):
	if env.gpu:
		return GPUReduce(env, dtype)
	else:
		return CPUReduce()

def createMaxFinder(env, dtype):
	if env.gpu:
		return GPUMaxFinder(env, dtype)
	else:
		return CPUMaxFinder()
