import numpy

from .misc import log2, PairedCalculation
from .transpose import createTranspose
from .typenames import MAP


class GPUReduce(PairedCalculation):

	def __init__(self, env, dtype, **kwds):
		PairedCalculation.__init__(self, env)
		self._tr = createTranspose(env, dtype)

		type = MAP[dtype]
		self._addParameters(operation="(a) + (b)", length=1, final_length=1, sparse=False,
			typename=type.name, dtype=type.dtype, double=type.precision.double)
		self.prepare(**kwds)

	def _prepare(self):
		assert self._p.length >= self._p.final_length, "Array size cannot be less than final size"
		assert self._p.length % self._p.final_length == 0

	def _gpu__prepare_specific(self):

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

			#define OP(a, b) (${p.operation})

			EXPORTED_FUNC void reduceKernel${block_size}(
				GLOBAL_MEM ${p.typename}* output, const GLOBAL_MEM ${p.typename}* input,
				int blocks_per_part, int last_block_size)
			{
				SHARED_MEM ${p.typename} shared_mem[${smem_size}];

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
					{
					%if p.typename.endswith('2'):
						shared_mem[tid].x = OP(shared_mem[tid].x,
							shared_mem[tid + ${2 ** reduction_pow}].x);
						shared_mem[tid].y = OP(shared_mem[tid].y,
							shared_mem[tid + ${2 ** reduction_pow}].y);
					%else:
						shared_mem[tid] = OP(shared_mem[tid],
							shared_mem[tid + ${2 ** reduction_pow}]);
					%endif
					}
					SYNC;
				%endfor

				// The following code will be executed inside a single warp, so no
				// shared memory synchronization is necessary
				%if log2_block_size > 0:
				if (tid < ${warp_size}) {
				#ifdef CUDA
				// Fix for Fermi videocards, see Compatibility Guide 1.2.2
				volatile ${p.typename} *smem = shared_mem;
				#else
				SHARED_MEM ${p.typename} *smem = shared_mem;
				#endif

				${p.typename} ttt;
				%for reduction_pow in xrange(min(log2_warp_size, log2_block_size - 1), -1, -1):
				%if p.typename.endswith('2'):
					ttt.x = OP(smem[tid].x, smem[tid + ${2 ** reduction_pow}].x);
					ttt.y = OP(smem[tid].y, smem[tid + ${2 ** reduction_pow}].y);
					smem[tid].x = ttt.x;
					smem[tid].y = ttt.y;
				%else:
					ttt = OP(smem[tid], smem[tid + ${2 ** reduction_pow}]);
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

		program = self._env.compile(kernel_template, p=self._p,
			warp_size=self._env.warp_size,
			double=self._p.double,
			max_block_size=self._env.max_block_size,
			log2=log2, block_sizes=block_sizes, reduce_powers=reduce_powers,
			construct_zero="0" if self._p.dtype in (numpy.float32, numpy.float64) else "complex_ctr(0, 0)")

		self._kernels = {}
		for block_size in block_sizes:
			name = "reduceKernel" + str(block_size)
			self._kernels[block_size] = getattr(program, name)

		# prepare call chain

		reduce_kernels = self._kernels
		self._prepared_calls = []

		# we can reduce maximum 'block size' times a pass
		max_reduce_power = self._max_block_size

		data_in = None
		length = self._p.length
		final_length = self._p.final_length

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

			grid_size = blocks_num * block_size

			if new_length == final_length:
				data_out = None
			else:
				data_out = self._env.allocate((new_length,), self._p.dtype)

			self._prepared_calls.append((
				reduce_kernels[block_size], (grid_size,), (block_size,),
					data_out, data_in,
					numpy.int32(blocks_per_part), numpy.int32(last_block_size)
			))

			length = new_length

			data_in = data_out

		if self._p.sparse:
			self._for_tr = self._env.allocate((self._p.length,), self._p.dtype)

	def __call__(self, data_in, data_out):
		if self._p.final_length == self._p.length:
			self._env.copyBuffer(data_in, dest=data_out)
			return

		if self._p.sparse:
			self._tr(data_in, self._for_tr, self._p.final_length,
				self._p.length / self._p.final_length)
			data_in = self._for_tr

		for func, grid, block, dout, din, bpp, lbs in self._prepared_calls:
			func.customCall(grid, block, data_out if dout is None else dout,
				data_in if din is None else din, bpp, lbs)


def max_func(x, axis=None):
	if axis is None:
		return x.real.max() + 1j * x.imag.max()
	else:
		return NotImplementedError()


class CPUReduce(PairedCalculation):

	def __init__(self, env, **kwds):
		PairedCalculation.__init__(self, env)

		self._addParameters(operation=numpy.sum, length=1, final_length=1, sparse=False,
			typename=None, dtype=None, double=True)
		self.prepare(**kwds)

	def __call__(self, data_in, data_out):

		data_in = data_in.ravel()[:self._p.length]

		if self._p.final_length == 1:
			data_out[0] = self._p.operation(data_in)
			return

		if self._p.sparse:
			data_in = data_in.reshape(self._p.length / self._p.final_length, self._p.final_length).T

		data_out.flat[:] = self._p.operation(
			data_in.reshape(self._p.final_length, self._p.length / self._p.final_length),
			axis=1).flat


def createReduce(env, dtype, **kwds):
	if env.gpu:
		return GPUReduce(env, dtype, **kwds)
	else:
		return CPUReduce(env, **kwds)

def createMaxFinder(env, dtype, **kwds):
	if env.gpu:
		return GPUReduce(env, dtype, operation="MAX((a), (b))", **kwds)
	else:
		return CPUReduce(env, operation=max_func, **kwds)
