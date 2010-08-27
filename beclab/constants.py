"""
Module, containing class with calculation constants
"""

import copy
import math
import numpy

from .globals import *

PSI_FUNC = 0
WIGNER = 1

COMP_1_minus1 = 0
COMP_2_1 = 1


class _Type:
	def __init__(self, name, dtype):
		self.name = name
		self.dtype = dtype
		self.nbytes = dtype().nbytes
		self.ctr = '(' + name + ')'
		self.cast = numpy.cast[dtype]

	def __getstate__(self):
		d = dict(self.__dict__)
		del d['cast']
		return d

	def __setstate__(self, state):
		self.__dict__ = state
		self.cast = numpy.cast[self.dtype]


class _Precision:
	def __init__(self, scalar, complex):
		self.scalar = scalar
		self.complex = complex

_single_float = _Type('float', numpy.float32)
_double_float = _Type('double', numpy.float64)

_single_complex = _Type('float2', numpy.complex64)
_double_complex = _Type('double2', numpy.complex128)

_single_precision = _Precision(_single_float, _single_complex)
_double_precision = _Precision(_double_float, _double_complex)


class Constants:
	"""Calculation constants, in natural units"""

	def __init__(self, model, double_precision=False):

		self.hbar = 1.054571628e-34 # Planck constant
		self.m = model.m
		a0 = 5.2917720859e-11 # Bohr radius

		model = copy.deepcopy(model)
		precision = _double_precision if double_precision else _single_precision
		self.scalar = precision.scalar
		self.complex = precision.complex

		self.nvx = model.nvx
		self.nvy = model.nvy
		self.nvz = model.nvz
		self.cells = self.nvx * self.nvy * self.nvz
		self.shape = (self.nvz, self.nvy, self.nvx)

		# prefix "w_" stands for radial frequency, "f_" for common frequency

		self.w_detuning = 2.0 * math.pi * model.detuning
		self.w_rabi = 2.0 * math.pi * model.rabi_freq
		self.t_rabi = 1.0 / model.rabi_freq

		# trap frequencies
		self.w_x = 2.0 * math.pi * model.fx
		self.w_y = 2.0 * math.pi * model.fy
		self.w_z = 2.0 * math.pi * model.fz

		self.gamma111 = model.gamma111
		self.gamma12 = model.gamma12
		self.gamma22 = model.gamma22

		g11 = 4.0 * math.pi * (self.hbar ** 2) * model.a11 * a0 / self.m
		g12 = 4.0 * math.pi * (self.hbar ** 2) * model.a12 * a0 / self.m
		g22 = 4.0 * math.pi * (self.hbar ** 2) * model.a22 * a0 / self.m

		self.g = {
			(COMP_1_minus1, COMP_1_minus1): g11,
			(COMP_1_minus1, COMP_2_1): g12,
			(COMP_2_1, COMP_1_minus1): g12,
			(COMP_2_1, COMP_2_1): g22
		}

		# g itself is too small for single precision
		self.g_by_hbar = dict((key, self.g[key] / self.hbar) for key in self.g)

		self.N = model.N

		mu1 = self.muTF(comp=COMP_1_minus1)

		self.e_cut = model.e_cut * mu1

		self.xmax = model.border * math.sqrt(2.0 * mu1 / (self.m * self.w_x ** 2))
		self.ymax = model.border * math.sqrt(2.0 * mu1 / (self.m * self.w_y ** 2))
		self.zmax = model.border * math.sqrt(2.0 * mu1 / (self.m * self.w_z ** 2))

		# space step
		self.dx = 2.0 * self.xmax / (self.nvx - 1)
		self.dy = 2.0 * self.ymax / (self.nvy - 1)
		self.dz = 2.0 * self.zmax / (self.nvz - 1)
		self.dV = self.dx * self.dy * self.dz
		self.V = self.dV * self.cells

		self.nvx_pow = log2(self.nvx)
		self.nvy_pow = log2(self.nvy)
		self.nvz_pow = log2(self.nvz)

		# k step
		self.dkx = math.pi / self.xmax
		self.dky = math.pi / self.ymax
		self.dkz = math.pi / self.zmax

		self.itmax = model.itmax
		self.dt_steady = model.dt_steady
		self.dt_evo = model.dt_evo
		self.ensembles = model.ensembles
		self.ens_shape = (self.ensembles * self.nvz, self.nvy, self.nvx)

		# cast all floating point values to current precision

		def recursiveCast(cast, obj):
			if isinstance(obj, dict):
				return dict([(key, recursiveCast(cast, obj[key])) for key in obj])
			elif isinstance(obj, list):
				return [recursiveCast(cast, elem) for elem in obj]
			elif isinstance(obj, float):
				return cast(obj)
			else:
				return obj

		# natural units
		self.t_rho = 1.0 / self.w_x
		self.e_rho = self.hbar * self.w_x
		self.l_rho = math.sqrt(self.hbar / (self.m * self.w_x))

		# By doing this, we can lose some small constants (they are turned into 0s)
		# So I decided to transform them in-place, only if it is necessary to pass
		# them in "real world" (opposed to using inside a template)
		# As a result, no more single precision for CPU (it worked even slower
		# than double precision anyway)
		#self.__dict__ = recursiveCast(self.scalar.cast, self.__dict__)

	def muTF(self, comp=COMP_1_minus1, N=None):
		"""get TF-approximated chemical potential"""
		if N is None:
			N = self.N

		g = self.g[(comp, comp)]

		w = (self.w_x * self.w_y * self.w_z) ** (1.0 / 3)
		return ((15 * N / (8.0 * math.pi)) ** 0.4) * \
			((self.m * w * w / 2) ** 0.6) * \
			(g ** 0.4)