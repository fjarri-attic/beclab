import numpy

from .typenames import single_precision, double_precision


class PyCUDARNG:

	def __init__(self, dtype):
		from pycuda.curandom import md5_code
		from pycuda.elementwise import get_elwise_kernel

		if dtype == numpy.complex64:
			self._func = get_elwise_kernel(
				"float2 *dest, unsigned int seed",
				md5_code + """
				#define POW_2_M32 (1/4294967296.0f)

				dest[i] = make_float2(a*POW_2_M32, b*POW_2_M32);
				if ((i += total_threads) < n)
					dest[i] = make_float2(c*POW_2_M32, d*POW_2_M32);
				""",
				"md5_rng_float")
		elif dtype == numpy.complex128:
			self._func = get_elwise_kernel(
				"pycuda::complex<double> *dest, unsigned int seed",
				md5_code + """
				#define POW_2_M32 (1/4294967296.0)
				#define POW_2_M64 (1/18446744073709551616.)

				dest[i] = pycuda::complex<double>(
					a*POW_2_M32 + b*POW_2_M64,
					c*POW_2_M32 + d*POW_2_M64);
				""",
				"md5_rng_float")

	def __call__(self, result, stream=None):
		self._func.prepared_async_call(result._grid, result._block, stream,
				result.gpudata, numpy.random.randint(2**31-1), result.size)


class CUDARandom:

	def __init__(self, env, double):
		self._env = env

		p = double_precision if double else single_precision
		self._scalar_dtype = p.scalar.dtype
		self._complex_dtype = p.complex.dtype

		self._scalar_cast = p.scalar.cast

		import pycuda.curandom as curandom
		#self._rand_func = curandom.rand
		self._rand_func = PyCUDARNG(self._complex_dtype)

		kernel_template = """
		<%!
			from math import pi
		%>

		EXPORTED_FUNC void randomNormal(int gsize,
			GLOBAL_MEM COMPLEX* data,
			SCALAR loc, SCALAR scale)
		{
			int id = GLOBAL_ID_FLAT;
			if(id >= gsize)
				return;

			COMPLEX u = data[id];
			SCALAR u1 = u.x, u2 = u.y;

			SCALAR ang = (SCALAR)${2.0 * pi} * u2;
			SCALAR c_ang = cos(ang);
			SCALAR s_ang = sin(ang);
			SCALAR coeff = sqrt(-(SCALAR)2.0 * log(u1)) * scale;

			data[id] = complex_ctr(coeff * c_ang + loc, coeff * s_ang + loc);
		}
		"""

		self._program = self._env.compile(kernel_template, double=double)
		self._randomNormalKernel = self._program.randomNormal

#	def rand(self, shape):
#		return self._rand_func(shape, dtype=self._scalar_dtype, stream=self._env.stream)

	def random_normal(self, result, scale=1.0, loc=0.0):
#		uniform = self.rand((2,) + result.shape)
		self._rand_func(result)
		self._randomNormalKernel(result.size, result,
			self._scalar_cast(loc), self._scalar_cast(scale / numpy.sqrt(2.0)))


class CurandRandom:

	def __init__(self, env, double):
		self._env = env

		kernel_template = """
		#include <curand_kernel.h>

		<%
			curand_normal2 = "curand_normal2" + ("" if c_ctype == 'float2' else "_double")
		%>

		extern "C" {

		__global__ void initialize(curandStateXORWOW *states, unsigned int *seeds)
		{
			const int idx = threadIdx.x + blockIdx.x * blockDim.x;
			curand_init(seeds[idx], idx, 0, &states[idx]);
		}

		__global__ void sample(curandStateXORWOW *states,
				${c_ctype} *randoms, ${s_ctype} scale, int size)
		{
			const int idx = threadIdx.x + blockIdx.x * blockDim.x;
			const int num_gens = blockDim.x * gridDim.x;
			int sample_idx = idx;
			curandStateXORWOW state = states[idx];

			while (sample_idx < size)
			{
				${c_ctype} rand_normal = ${curand_normal2}(&state);
				rand_normal.x *= scale;
				rand_normal.y *= scale;

				randoms[sample_idx] = rand_normal;

				sample_idx += num_gens;
			}
		}

		}
		"""

		p = double_precision if double else single_precision
		self._scalar_cast = p.scalar.cast
		self._program = self._env.compile(kernel_template,
			manual_extern_c=True, c_ctype=p.complex.name, s_ctype=p.scalar.name)
		self._initialize = self._program.initialize
		self._sample = self._program.sample

		self.seed()

	def seed(self, seed=None):
		from pycuda.characterize import sizeof, has_stack
		import pycuda.driver as cuda
		import pycuda.gpuarray as gpuarray

		rng = numpy.random.RandomState()
		rng.seed(seed)

		gen_block_size = min(
			self._initialize.max_threads_per_block,
			self._sample.max_threads_per_block)
		gen_grid_size = self._env.device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
		gen_block = (gen_block_size, 1, 1)
		gen_gsize = (gen_grid_size * gen_block_size, 1, 1)

		num_gen = gen_block_size * gen_grid_size
		assert num_gen <= 20000

		seeds = gpuarray.to_gpu(rng.randint(0, 2**32 - 1, size=num_gen).astype(numpy.uint32))
		state_type_size = sizeof("curandStateXORWOW", "#include <curand_kernel.h>")
		self.states = gpuarray.GPUArray(num_gen * state_type_size, numpy.uint8)
		self._initialize.customCall(gen_gsize, gen_block, self.states.gpudata, seeds.gpudata)
		self._env.synchronize()
		self.gsize = gen_gsize
		self.lsize = gen_block

	def get_state(self):
		return self.states.get()

	def set_state(self, states):
		import pycuda.gpuarray as gpuarray
		self._env.synchronize()
		self.states = self._env.toDevice(states)

	def random_normal(self, result, scale=1.0):
		self._sample.customCall(self.gsize, self.lsize,
			self.states, result,
			self._scalar_cast(scale / numpy.sqrt(2.0)), numpy.int32(result.size))


class NewCUDARandom:

	def __init__(self, env, double):
		self._env = env

		p = double_precision if double else single_precision
		self._scalar_dtype = p.scalar.dtype
		self._complex_dtype = p.complex.dtype

		self._scalar_cast = p.scalar.cast
		self._complex_cast = p.complex.cast

		from pycuda.curandom import XORWOWRandomNumberGenerator as RNG
		self._rng = RNG()

		kernel_template = """
		<%!
			from math import pi
		%>

		EXPORTED_FUNC void scaleRandoms(int gsize,
			GLOBAL_MEM COMPLEX* data,
			COMPLEX loc, SCALAR scale)
		{
			int id = GLOBAL_ID_FLAT;
			if(id >= gsize)
				return;

			COMPLEX r = data[id];
			r.x += loc.x;
			r.y += loc.y;
			r.x *= scale;
			r.y *= scale;
			data[id] = r;
		}
		"""

		self._program = self._env.compile(kernel_template, double=double)
		self._scaleRandoms = self._program.scaleRandoms

	def random_normal(self, result, scale=1.0, loc=0.0):
		self._rng.fill_normal(result, stream=self._env.stream)
		self._scaleRandoms(result.size, result,
			self._complex_cast(loc), self._scalar_cast(scale / numpy.sqrt(2.0)))


class FakeRandom:
	# FIXME: temporary stub. Have to write proper GPU-based generator

	def __init__(self, env, double):
		self._env = env
		self._random = CPURandom(env, double)

	def rand(self, shape):
		return self._env.toDevice(self._random.rand(shape))

	def random_normal(self, size, scale=1.0, loc=0.0):
		return self._env.toDevice(self._random.random_normal(size, scale=scale / numpy.sqrt(2.0), loc=loc))


class CPURandom:

	def __init__(self, env, double):
		p = double_precision if double else single_precision
		self._scalar_dtype = p.scalar.dtype
		self._complex_dtype = p.complex.dtype
		self._env = env

	def rand(self, shape):
		return numpy.random.rand(*shape).astype(self._scalar_dtype)

	def random_normal(self, result, scale=1.0, loc=0.0):
		complex_scale = scale / numpy.sqrt(2.0)
		randoms = (
			numpy.random.normal(loc=loc, scale=complex_scale, size=result.shape) +
			1j * numpy.random.normal(loc=loc, scale=complex_scale, size=result.shape)
		).astype(self._complex_dtype)
		self._env.copyBuffer(randoms, dest=result)


def createRandom(env, double):
	if env.gpu and env.cuda:
		return CurandRandom(env, double)
	elif env.gpu:
		return FakeRandom(env, double)
	else:
		return CPURandom(env, double)
