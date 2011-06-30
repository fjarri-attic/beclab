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
			dest.flat[:] = buf.flat

	def __str__(self):
		return "CPU"

	def release(self):
		pass

	def compile(self, source, constants, **kwds):
		raise NotImplementedError("compile() called for CPU environment")
