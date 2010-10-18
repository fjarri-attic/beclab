import numpy

from .typenames import single_precision, double_precision


class CUDARandom:

	def __init__(self, env, double):
		self._env = env

		p = double_precision if double else single_precision
		self._scalar_dtype = p.scalar.dtype
		self._complex_dtype = p.complex.dtype

		self._scalar_cast = p.scalar.cast

		import pycuda.curandom as curandom
		self._rand_func = curandom.rand

		kernel_template = """
		<%!
			from math import pi
		%>

		EXPORTED_FUNC void randomNormal(
			GLOBAL_MEM COMPLEX* normal, const GLOBAL_MEM COMPLEX* uniform,
			SCALAR loc, SCALAR scale)
		{
			int id = GLOBAL_ID_FLAT;

			COMPLEX u = uniform[id];
			SCALAR u1 = u.x, u2 = u.y;

			SCALAR ang = (SCALAR)${2.0 * pi} * u2;
			SCALAR c_ang = cos(ang);
			SCALAR s_ang = sin(ang);
			SCALAR coeff = sqrt(-(SCALAR)2.0 * log(u1)) * scale;

			normal[id] = complex_ctr(coeff * c_ang + loc, coeff * s_ang + loc);
		}
		"""

		self._program = self._env.compile(kernel_template, double=double)
		self._randomNormalKernel = self._program.randomNormal

	def rand(self, shape):
		return self._rand_func(shape, dtype=self._scalar_dtype, stream=self._env.stream)

	def random_normal(self, size, scale=1.0, loc=0.0):
		uniform = self.rand((size * 2,))
		normal = self._env.allocate((size,), self._complex_dtype)
		self._randomNormalKernel(size, normal, uniform, self._scalar_cast(loc), self._scalar_cast(scale))
		return normal


class FakeRandom:
	# FIXME: temporary stub. Have to write proper GPU-based generator

	def __init__(self, env, double):
		self._env = env
		self._random = CPURandom(env, double)

	def rand(self, shape):
		return self._env.toDevice(self._random.rand(shape))

	def random_normal(self, size, scale=1.0, loc=0.0):
		return self._env.toDevice(self._random.random_normal(size, scale=scale, loc=loc))


class CPURandom:

	def __init__(self, env, double):
		p = double_precision if double else single_precision
		self._scalar_dtype = p.scalar.dtype
		self._complex_dtype = p.complex.dtype

	def rand(self, shape):
		return numpy.random.rand(*shape).astype(self._scalar_dtype)

	def random_normal(self, size, scale=1.0, loc=0.0):
		return (numpy.random.normal(loc=loc, scale=scale, size=size) +
			1j * numpy.random.normal(loc=loc, scale=scale, size=size)).astype(self._complex_dtype)


def createRandom(env, double):
	if env.gpu and env.cuda:
		return CUDARandom(env, double)
	elif env.gpu:
		return FakeRandom(env, double)
	else:
		return CPURandom(env, double)
