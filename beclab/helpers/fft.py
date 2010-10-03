import numpy


class NumpyPlan3D:

	def __init__(self, main_dim):
		self._main_dim = main_dim

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifftn
		else:
			func = numpy.fft.fftn

		for i in xrange(batch):
			start = i * self._main_dim
			stop = (i + 1) * self._main_dim
			data_out[start:stop,:,:] = func(data_in[start:stop,:,:])


class NumpyPlan2D:

	def __init__(self, main_dim):
		self._main_dim = main_dim

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifftn
		else:
			func = numpy.fft.fftn

		for i in xrange(batch):
			start = i * self._main_dim
			stop = (i + 1) * self._main_dim
			data_out[start:stop,:] = func(data_in[start:stop,:])


class NumpyPlan1D:

	def __init__(self, main_dim):
		self._main_dim = main_dim

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifft
		else:
			func = numpy.fft.fft

		for i in xrange(batch):
			start = i * self._main_dim
			stop = (i + 1) * self._main_dim
			data_out[start:stop] = func(data_in[start:stop])


def createFFTPlan(env, shape, dtype):

	if env.gpu:
		if env.cuda:
			import pyfft.cuda
			return pyfft.cuda.Plan(shape, dtype=dtype, normalize=True,
				stream=env.stream, context=env.context)
		else:
			import pyfft.cl
			return pyfft.cl.Plan(shape, dtype=dtype, normalize=True, queue=env.queue)
	else:
		plans = {
			1: NumpyPlan1D,
			2: NumpyPlan2D,
			3: NumpyPlan3D
		}

		if len(shape) not in plans:
			raise ValueError("Wrong dimension")

		return plans[len(shape)](shape[0])
