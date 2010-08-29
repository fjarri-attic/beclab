try:
	import pyfft.cl
except:
	pass

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


def createPlan(env, constants, shape):
	if env.gpu:
		return pyfft.cl.Plan(shape, dtype=constants.complex.dtype, normalize=True, queue=env.queue)
	else:
		if len(shape) == 3:
			return NumpyPlan3D(shape[0])
		else:
			return NumpyPlan1D(shape[0])
