import numpy
from mako.template import Template

try:
	import pyopencl as cl
except:
	pass

_kernel_template = Template("""
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
__kernel void transposeKernel(__global ${typename}* odata, const __global ${typename}* idata,
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
	__local ${typename} block[(${half_warp_size} + 1) * ${half_warp_size}];

	unsigned int lid_x = get_local_id(0);
	unsigned int lid_y = get_local_id(1);

	unsigned int gid_x = get_group_id(0);
	unsigned int gid_y = get_group_id(1);

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

		barrier(CLK_LOCAL_MEM_FENCE);

		odata[index_out] = block[index_transpose];

		index_in += size;
		index_out += size;
	}
}
""")

class Transpose:

	def __init__(self, env, type):

		self._half_warp_size = 16
		self._queue = env.queue

		# render function from template
		source = _kernel_template.render(typename=type.name, half_warp_size=self._half_warp_size)

		# get function from module
		_kernel_module = cl.Program(env.context, source).build()
		self._func = _kernel_module.transposeKernel
		self._local_size = (self._half_warp_size, self._half_warp_size)

	def __call__(self, odata, idata, width, height, num=1):
		"""
		Fast matrix transpose function
		odata: Output buffer for transposed batch of matrices, must be different than idata
		idata: Input batch of matrices
		width: Width of each matrix, must be a multiple of HALF_WARP_SIZE
		height: Height of each matrix, must be a multiple of HALF_WARP_SIZE
		num: number of matrices in the batch
		"""
		assert width % self._half_warp_size == 0
		assert height % self._half_warp_size == 0

		global_size = (width, height)
		self._func(self._queue, global_size, odata, idata, numpy.int32(width),
			numpy.int32(height), numpy.int32(num), local_size=self._local_size)
