import numpy

class _Type:
	def __init__(self, name, dtype):
		self.name = name
		self.dtype = dtype
		self.nbytes = dtype().nbytes
		self.cast = dtype # numpy.cast[dtype]
		self.is_complex = (dtype in (numpy.complex64, numpy.complex128))

	def __getstate__(self):
		d = dict(self.__dict__)
		del d['cast']
		return d

	def __setstate__(self, state):
		self.__dict__ = state
		self.cast = numpy.cast[self.dtype]


class _Precision:
	def __init__(self, scalar, complex, double):
		self.scalar = scalar
		self.scalar.precision = self
		self.complex = complex
		self.complex.precision = self
		self.double = double

_single_scalar = _Type('float', numpy.float32)
_double_scalar = _Type('double', numpy.float64)

_single_complex = _Type('float2', numpy.complex64)
_double_complex = _Type('double2', numpy.complex128)

single_precision = _Precision(_single_scalar, _single_complex, False)
double_precision = _Precision(_double_scalar, _double_complex, True)

MAP = {
	numpy.float32: _single_scalar,
	numpy.float64: _double_scalar,
	numpy.complex64: _single_complex,
	numpy.complex128: _double_complex
}
