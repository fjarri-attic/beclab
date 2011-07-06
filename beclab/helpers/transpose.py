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
		if(xIndex < width && yIndex < height)
			block[index_block] = idata[index_in];

		SYNC;

		if(xBlock + lid_y < width && yBlock + lid_x < height)
			odata[index_out] = block[index_transpose];

		index_in += size;
		index_out += size;
	}
}
"""

_permute_template = """
EXPORTED_FUNC void  permuteKernel(${typename} *odata, ${typename} *idata,
    unsigned int n1, unsigned int n2, unsigned int n3,
    unsigned int Gx, unsigned int Gz,
    unsigned int k2, unsigned int batch)
{
    __shared__ ${typename} block[${half_warp_size}][${half_warp_size + 1}];

    unsigned int s1, s2, t1, t2;
    unsigned int xIndex_in, yIndex, zIndex_in, xIndex_out, zIndex_out;
    unsigned int index_in, index_out;

    // step 1: transform grid index to 3D corase grid index
    //  blockIdx.x = Gz * s1 + t1
    //  blockIdx.y = Gx * s2 + t2
    //
    //  where (s1, s2): index to y-direction, (t1, t2): index to x-z slice (local blockID )
	s1 = BLOCK_ID_X / Gz;
	t1 = BLOCK_ID_X % Gz;

    s2 = BLOCK_ID_Y / Gx;
    t2 = BLOCK_ID_Y % Gx;

    // step 2: yIndex = s2*k2 + s1 from (s1, s2)
    yIndex = s2 * k2 + s1;
	if(yIndex >= n2)
		return;

    zIndex_in = t1 * ${half_warp_size} + THREAD_ID_X;
    xIndex_in = t2 * ${half_warp_size} + THREAD_ID_Y;
    index_in = (xIndex_in * n2 + yIndex) * n3 + zIndex_in;

    xIndex_out = t2 * ${half_warp_size} + THREAD_ID_X;
    zIndex_out = t1 * ${half_warp_size} + THREAD_ID_Y;
    index_out = (yIndex * n3 + zIndex_out) * n1 + xIndex_out;

	int size = n1 * n2 * n3;

	// FIXME: probably it will be faster to make more blocks instead of
	// doing a loop.

	for(int i = 0; i < batch; i++)
	{
    	// step 3: read the matrix tile into shared memory
	    if((xIndex_in < n1) && (zIndex_in < n3))
	    {
	        block[THREAD_ID_Y][THREAD_ID_X] = idata[index_in];
	    }
	    __syncthreads();

	    // step 4: write the transposed matrix tile to global memory
	    if((xIndex_out < n1) && (zIndex_out < n3))
	    {
	        odata[index_out] = block[THREAD_ID_X][THREAD_ID_Y];
	    }
	    __syncthreads();

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
		full_width = ((width - 1) / self._half_warp_size + 1) * self._half_warp_size
		full_height = ((height - 1) / self._half_warp_size + 1) * self._half_warp_size

		global_size = (full_width, full_height)
		self._func.customCall(global_size, self._local_size,
			odata, idata, numpy.int32(width),
			numpy.int32(height), numpy.int32(batch))


class GPUPermute:

	def __init__(self, env, dtype):

		self._half_warp_size = env.warp_size / 2

		type = MAP[dtype]

		self._program = env.compile(_permute_template,
			double=type.precision.double, typename=type.name,
			half_warp_size=self._half_warp_size)
		self._func = self._program.permuteKernel

	def __call__(self, src, dest, src_shape, batch=1):
		n1, n2, n3 = src_shape[-3:]

		# Gx = number of grids need in x-axis
		# Gz = number of grids need in z-axis
		#
		# we call a coarse grid is compose of grid Gx x Gz
		Gx = (n1 + self._half_warp_size - 1) / self._half_warp_size
		Gz = (n3 + self._half_warp_size - 1) / self._half_warp_size

		# since a coarse can cover a x-z slice, we need n2 corase grids to cover X
		# in order to save resource, we want to find two integers k1, k2 such that
		#       k1 * k2 - n2 <= 1
		#  for example:
		#       n2 = 7   ==> k1 = 2 and k2 = 4
		#       n2 = 13  ==> k2 = 2 and k2 = 7
		db_n2 = float(n2)
		max_k1 = int(numpy.sqrt(db_n2))

		for k1 in xrange(max_k1, 0, -1):
			pass
			k2 = int(db_n2 / float(k1)) + 1
			if k1 * k2 - n2 <= 1:
				break

		block = (self._half_warp_size, self._half_warp_size)
		global_size = (k2 * Gz * self._half_warp_size, k1 * Gx * self._half_warp_size)

		int_cast = numpy.int32

		self._func.customCall(global_size, block,
			dest, src, int_cast(n1), int_cast(n2), int_cast(n3), int_cast(Gx), int_cast(Gz),
			int_cast(k2), int_cast(batch))


class CPUTranspose:

	def __call__(self, idata, odata, width, height, batch=1):
		idata = idata.ravel()
		odata = odata.ravel()
		for i in xrange(batch):
			start = i * width * height
			stop = (i + 1) * width * height
			odata[start:stop] = idata[start:stop].reshape((height, width)).T.ravel()


class CPUPermute:

	def __call__(self, src, dest, src_shape, batch=1):
		shape = tuple(numpy.arange(len(src_shape)))
		axes = shape[:-3] + (shape[-2], shape[-1], shape[-3])
		dest.flat[:] = (numpy.transpose(src, axes=axes)).flat

def createTranspose(env, dtype):
	if env.gpu:
		return GPUTranspose(env, dtype)
	else:
		return CPUTranspose()

def createPermute(env, dtype):
	if env.gpu:
		return GPUPermute(env, dtype)
	else:
		return CPUPermute()
