import numpy
import gc


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

	def copyBuffer(self, buf, dest, src_offset=0, dest_offset=0, length=None):
		size = buf.size if length is None else length
		dest.flat[src_offset:src_offset + size] = buf.flat[dest_offset:dest_offset + size]

	def __str__(self):
		return "CPU"

	def release(self):
		gc.collect() # forcefully frees all buffers

	def compile(self, source, constants, **kwds):
		raise NotImplementedError("compile() called for CPU environment")

	def supportsDouble(self):
		return True
