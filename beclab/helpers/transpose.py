import numpy
from mako.template import Template

from .typenames import MAP


_kernel_template = """
/**
 * Fast matrix transpose kernel
 *
 * Uses shared memory to coalesce global memory reads and writes, improving performance.
 *
 * @param odata Output buffer for transposed batch of matrices, must be different than idata
 * @param idata Input batch of matrices
 * @param width Width of each matrix, must be a multiple of HALF_WARP_SIZE
 * @param height Height of each matrix, must be a multiple of HALF_WARP_SIZE
 * @param num Matrices in the batch
 */
EXPORTED_FUNC void transposeKernel(GLOBAL_MEM ${typename}* odata, const GLOBAL_MEM ${typename}* idata,
	unsigned int width, unsigned int height, unsigned int num)
{
	// To prevent shared memory bank confilcts:
	// - Load each component into a different array. Since the array size is a
	//   multiple of the number of banks (16), each thread reads x and y from
	//   the same bank. If a single value_pair array is used, thread n would read
	//   x and y from banks n and n+1, and thread n+8 would read values from the
	//   same banks - causing a bank conflict.
	// - Use HALF_WARP_SIZE+1 as the x size of the array. This way each row of the
	//   array starts in a different bank - so reading from shared memory
	//   doesn't cause bank conflicts when writing the transpose out to global
	//   memory.
	SHARED_MEM ${typename} block[(${half_warp_size} + 1) * ${half_warp_size}];

	unsigned int lid_x = THREAD_ID_X;
	unsigned int lid_y = THREAD_ID_Y;

	unsigned int gid_x = BLOCK_ID_X;
	unsigned int gid_y = BLOCK_ID_Y;

	const unsigned int half_warp_size = ${half_warp_size};

	unsigned int xBlock = mul24(half_warp_size, gid_x);
	unsigned int yBlock = mul24(half_warp_size, gid_y);
	unsigned int xIndex = xBlock + lid_x;
	unsigned int yIndex = yBlock + lid_y;
	unsigned int size = mul24(width, height);
	unsigned int index_block = mul24(lid_y, half_warp_size + 1) + lid_x;
	unsigned int index_transpose = mul24(lid_x, half_warp_size + 1) + lid_y;
	unsigned int index_in = mul24(width, yIndex) + xIndex;
	unsigned int index_out = mul24(height, xBlock + lid_y) + yBlock + lid_x;

	for(int n = 0; n < num; ++n)
	{
		block[index_block] = idata[index_in];

		SYNC;

		odata[index_out] = block[index_transpose];

		index_in += size;
		index_out += size;
	}
}
"""

class GPUTranspose:

	def __init__(self, env, dtype):

		self._half_warp_size = env.warp_size / 2

		type = MAP[dtype]

		self._program = env.compile(_kernel_template,
			double=type.precision.double, typename=type.name,
			half_warp_size=self._half_warp_size)
		self._func = self._program.transposeKernel
		self._local_size = (self._half_warp_size, self._half_warp_size)

	def __call__(self, idata, odata, width, height, batch=1):
		"""
		Fast matrix transpose function
		idata: Input batch of matrices
		odata: Output buffer for transposed batch of matrices, must be different than idata
		width: Width of each matrix, must be a multiple of HALF_WARP_SIZE
		height: Height of each matrix, must be a multiple of HALF_WARP_SIZE
		num: number of matrices in the batch
		"""
		assert width % self._half_warp_size == 0
		assert height % self._half_warp_size == 0

		global_size = (width, height)
		self._func.customCall(global_size, self._local_size,
			odata, idata, numpy.int32(width),
			numpy.int32(height), numpy.int32(batch))


class CPUTranspose:

	def __call__(self, idata, odata, width, height, batch=1):
		idata = idata.ravel()
		odata = odata.ravel()
		for i in xrange(batch):
			start = i * width * height
			stop = (i + 1) * width * height
			odata[start:stop] = idata[start:stop].reshape((height, width)).T.ravel()


def createTranspose(env, dtype):
	if env.gpu:
		return GPUTranspose(env, dtype)
	else:
		return CPUTranspose()
