import numpy


class CPUEnvironment:

	def __init__(self):
		self.gpu = False

	def allocate(self, shape, dtype):
		return numpy.empty(shape, dtype=dtype)

	def synchronize(self):
		pass

	def fromDevice(self, buf, shape=None):
		buf = buf.copy()
		return buf if shape is None else buf.reshape(shape)

	def toDevice(self, buf, shape=None):
		buf = buf.copy()
		return buf if shape is None else buf.reshape(shape)

	def copyBuffer(self, buf, dest=None):
		if dest is None:
			return buf.copy()
		else:
			# FIXME: numpy does not seem to have special function
			# for filling array with other array
			if len(buf.shape) == 3:
				dest[:,:,:] = buf
			elif len(buf.shape) == 2:
				dest[:,:] = buf
			else:
				dest[:] = buf

	def __str__(self):
		return "CPU"

	def release(self):
		pass

	def compile(self, source, constants, **kwds):
		raise NotImplementedError("compile() called for CPU environment")
