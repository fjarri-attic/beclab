import numpy

from .misc import PairedCalculation
from .transpose import createTranspose
from .typenames import MAP


dot_template = """
	<%
		if type_a.is_complex and type_b.is_complex:
			matrix_mul_func = 'complex_mul'
			scale_mul_func = 'complex_mul_scalar'
		elif type_a.is_complex and not type_b.is_complex:
			matrix_mul_func = 'complex_mul_scalar'
			scale_mul_func = 'complex_mul_scalar'
		else:
			matrix_mul_func = 'scalar_mul'
			scale_mul_func = 'scalar_mul'

		get_zero_ctr = lambda t: 'complex_ctr(0, 0)' if t.is_complex else '0'
		zero_ctr_a = get_zero_ctr(type_a)
		zero_ctr_b = get_zero_ctr(type_b)
		zero_ctr_res = get_zero_ctr(type_res)
	%>
	INTERNAL_FUNC SCALAR scalar_mul(SCALAR a, SCALAR b)
	{
		return a * b;
	}

	EXPORTED_FUNC void dot_batched(GLOBAL_MEM ${type_res.name}* C,
		GLOBAL_MEM ${type_a.name}* A, GLOBAL_MEM ${type_b.name}* B,
		SCALAR scale, int hA, int wA, int wB, int blocks_per_matrix)
	{
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ ${type_a.name} As[${block_size}][${block_size}];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ ${type_b.name} Bs[${block_size}][${block_size}];

		// Block index
		int bx = blockIdx.x;
		int by = blockIdx.y;

		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		int matrix_num = by / blocks_per_matrix;
		by = by - blocks_per_matrix * matrix_num;

		A += matrix_num * hA * wA;
		%if batched_b:
			B += matrix_num * wA * wB;
		%endif
		C += matrix_num * hA * wB;

		// Index of the first sub-matrix of A processed by the block
		int aBegin = wA * ${block_size} * by;

		// Index of the last sub-matrix of A processed by the block
		int aEnd   = aBegin + wA - 1;

		// Step size used to iterate through the sub-matrices of A
		int aStep  = ${block_size};

		// Index of the first sub-matrix of B processed by the block
		int bBegin = ${block_size} * bx;

		// Step size used to iterate through the sub-matrices of B
		int bStep  = ${block_size} * wB;

		// Csub is used to store the element of the block sub-matrix
		// that is computed by the thread
		${type_res.name} Csub = ${zero_ctr_res};

		// Loop over all the sub-matrices of A and B
		// required to compute the block sub-matrix
		for (int a = aBegin, b = bBegin, step = 0;
				a <= aEnd;
				a += aStep, b += bStep, step++) {

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			int a_x = step * ${block_size} + tx;
			int a_y = by * ${block_size} + ty;
			int b_x = bx * ${block_size} + tx;
			int b_y = step * ${block_size} + ty;

			As[ty][tx] = (a_x < wA && a_y < hA) ? A[a + wA * ty + tx] : ${zero_ctr_a};
			Bs[ty][tx] = (b_x < wB && b_y < wA) ? B[b + wB * ty + tx] : ${zero_ctr_b};

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix

			%for k in xrange(block_size):
				Csub = Csub + ${matrix_mul_func}(As[ty][${k}], Bs[${k}][tx]);
			%endfor

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}

		// Write the block sub-matrix to device memory;
		// each thread writes one element
		int c_x = ${block_size} * bx + tx;
		int c_y = ${block_size} * by + ty;
		if(c_y < hA && c_x < wB)
			C[wB * c_y + c_x] = ${scale_mul_func}(Csub, scale);
	}
"""

def min_blocks(length, block):
	return (length - 1) / block + 1


class GPUDot(PairedCalculation):

	def __init__(self, env, dtype_a, dtype_b, dtype_res, dtype_scale, batched_b):
		PairedCalculation.__init__(self, env)
		self.block_size = env.warp_size / 2

		type_a = MAP[dtype_a]
		type_b = MAP[dtype_b]
		type_res = MAP[dtype_res]

		self._type_scale = MAP[dtype_scale]
		self._batched_b = batched_b

		self._program = env.compile(dot_template,
			type_a=type_a, type_b=type_b, type_res=type_res,
			block_size=self.block_size,
			batched_b=batched_b,
			double=(dtype_a in (numpy.float64, numpy.complex128)))
		self.kernel = self._program.dot_batched

	def __call__(self, res, a, b, ha, wa, wb, batch, scale=1):
		#assert a.size == batch * ha * wa
		#assert b.size == (batch if self._batched_b else 1) * wa * wb
		#assert res.size == batch * ha * wb
		block_size = self.block_size
		blocks_per_matrix = min_blocks(ha, block_size)

		self.kernel._customCall(
			(
				min_blocks(wb, block_size),
				blocks_per_matrix * batch
			),
			(block_size, block_size, 1),
			res, a, b,
			self._type_scale.cast(scale),
			numpy.int32(ha),
			numpy.int32(wa), numpy.int32(wb),
			numpy.int32(blocks_per_matrix),
		)


class CPUDot:

	def __init__(self, batched_b):
		self._batched_b = batched_b

	def __call__(self, res, a, b, ha, wa, wb, batch, scale=1):
		a_view = a.ravel()[:ha*wa*batch].reshape(batch, ha, wa)
		if self._batched_b:
			b_view = b.ravel()[:wa*wb*batch].reshape(batch, wa, wb)
		else:
			b_view = b.ravel()[:wa*wb].reshape(wa, wb)

		res_view = res.ravel()[:ha*wb*batch].reshape(batch, ha, wb)
		for i in xrange(batch):
			b = b_view[i] if self._batched_b else b_view
			res_view[i].flat[:] = numpy.dot(a_view[i], b).flat
		res_view *= scale


def createDot(env, dtype_a, dtype_b, dtype_res, dtype_scale, batched_b):
	if env.gpu:
		return GPUDot(env, dtype_a, dtype_b, dtype_res, dtype_scale, batched_b)
	else:
		return CPUDot(batched_b)
