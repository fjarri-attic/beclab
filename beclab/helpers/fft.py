import numpy


class NumpyPlan3D:

	def __init__(self, shape, scale):
		self._shape = shape
		self._scale_coeff = numpy.sqrt(scale[0] * scale[1] * scale[2] /
			(shape[0] * shape[1] * shape[2]))

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifftn
			coeff = 1.0 / self._scale_coeff
		else:
			func = numpy.fft.fftn
			coeff = self._scale_coeff

		shape = self._shape
		data_out.flat[:] = func(data_in.reshape(batch, *shape), axes=(1, 2, 3)).flat
		data_out *= coeff


class NumpyPlan1D:

	def __init__(self, shape, scale):
		self._shape = shape
		self._scale_coeff = numpy.sqrt(scale[0] / shape[0])

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifft
			coeff = 1.0 / self._scale_coeff
		else:
			func = numpy.fft.fft
			coeff = self._scale_coeff

		shape = self._shape
		data_out.flat[:] = func(data_in.reshape(batch, *shape)).flat
		data_out *= coeff


def createFFTPlan(env, constants, grid):

	shape = grid.shape
	dtype = constants.complex.dtype
	if grid.dim == 1:
		scale = (grid.dz,)
	else:
		scale = (grid.dz, grid.dy, grid.dx)

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
