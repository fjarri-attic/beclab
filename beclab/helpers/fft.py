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
		data_out.flat[:] = func(data_in.reshape(batch, *shape), axes=(1, 2, 3)).flat
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


def createFFTPlan(env, constants, grid):

	shape = grid.shape
	dtype = constants.complex.dtype
	if grid.dim == 1:
		scale = numpy.sqrt(grid.dz / grid.shape[0])
	else:
		scale = numpy.sqrt(grid.dz * grid.dy * grid.dx / grid.size)

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
			3: NumpyPlan3D
		}

		if len(shape) not in plans:
			raise ValueError("Wrong dimension")

		return plans[len(shape)](shape, scale)
