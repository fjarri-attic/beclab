try:
	import pyfft.cl
except:
	pass

import numpy

class NumpyPlan:

	def __init__(self, x, y, z):
		self._x = x
		self._y = y
		self._z = z

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		if data_out is None:
			data_out = data_in

		if inverse:
			func = numpy.fft.ifftn
		else:
			func = numpy.fft.fftn

		for i in xrange(batch):
			start = i * self._z
			stop = (i + 1) * self._z
			data_out[start:stop,:,:] = func(data_in[start:stop,:,:])

def createPlan(env, constants, x, y, z):
	if env.gpu:
		return pyfft.cl.Plan((z, y, x), dtype=constants.complex.dtype, normalize=True, queue=env.queue)
	else:
		return NumpyPlan(x, y, z)
