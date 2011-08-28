import numpy


class NumpyPlan3D:

	def __init__(self, shape, scale):
		self._shape = shape
		self._scale = scale

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifftn
			coeff = 1.0 / self._scale
		else:
			func = numpy.fft.fftn
			coeff = self._scale

		shape = self._shape
		data_out.flat[:] = func(data_in.reshape(batch, *shape), axes=(-3, -2, -1)).flat
		data_out *= coeff


class NumpyPlan2D:

	def __init__(self, shape, scale):
		self._shape = shape
		self._scale = scale

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifftn
			coeff = 1.0 / self._scale
		else:
			func = numpy.fft.fftn
			coeff = self._scale

		shape = self._shape
		data_out.flat[:] = func(data_in.reshape(batch, *shape), axes=(-2, -1)).flat
		data_out *= coeff


class NumpyPlan1D:

	def __init__(self, shape, scale):
		self._shape = shape
		self._scale = scale

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifft
			coeff = 1.0 / self._scale
		else:
			func = numpy.fft.fft
			coeff = self._scale

		shape = self._shape
		data_out.flat[:] = func(data_in.reshape(batch, *shape)).flat
		data_out *= coeff

class FakeGPUFFT1D:

	def __init__(self, env, shape, scale):
		self._shape = shape
		self._scale = scale
		self._env = env

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifft
			coeff = 1.0 / self._scale
		else:
			func = numpy.fft.fft
			coeff = self._scale

		d = self._env.fromDevice(data_in)
		shape = self._shape
		dout = func(d.reshape(batch, *shape))
		dout *= coeff
		dout_gpu = self._env.toDevice(dout)
		self._env.copyBuffer(dout_gpu, dest=data_out)


def createFFTPlan(env, constants, grid):

	shape = grid.shape
	dtype = constants.complex.dtype
	scale = numpy.sqrt(grid.dV_uniform / grid.size)

	if env.gpu:
		if env.cuda:
			import pyfft.cuda
			return pyfft.cuda.Plan(shape, dtype=dtype, normalize=True,
				stream=env.stream, context=env.context, scale=scale)
		else:
			import pyfft.cl
			return pyfft.cl.Plan(shape, dtype=dtype, normalize=True,
				queue=env.queue, scale=scale)
	else:
		plans = {
			1: NumpyPlan1D,
			2: NumpyPlan2D,
			3: NumpyPlan3D
		}

		if len(shape) not in plans:
			raise ValueError("Wrong dimension")

		return plans[len(shape)](shape, scale)
