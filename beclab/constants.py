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
		model = copy.deepcopy(model)
		precision = _double_precision if double_precision else _single_precision
		self.scalar = precision.scalar
		self.complex = precision.complex

		w_rho = 2.0 * math.pi * model.fx # radial oscillator frequency
		l_rho = math.sqrt(model.hbar / (model.m * w_rho)) # natural length
		self.l_rho = l_rho
		self.lambda_ = model.fx / model.fz
		self.w_rho = w_rho

		self.t_rho = 1.0 / w_rho # natural time unit

		self.nvx = model.nvx
		self.nvy = model.nvy
		self.nvz = model.nvz
		self.cells = self.nvx * self.nvy * self.nvz
		self.shape = (self.nvz, self.nvy, self.nvx)

		self.detuning = 2 * math.pi * model.detuning / w_rho
		self.rabi_freq = 2 * math.pi * model.rabi_freq / w_rho
		self.rabi_period = 1.0 / self.rabi_freq

		self.l111 = model.gamma111 / (pow(l_rho, 6) * w_rho)
		self.l12 = model.gamma12 / (pow(l_rho, 3) * w_rho)
		self.l22 = model.gamma22 / (pow(l_rho, 3) * w_rho)

		self.g11 = 4 * math.pi * model.a11 * model.a0 / l_rho
		self.g12 = 4 * math.pi * model.a12 * model.a0 / l_rho
		self.g22 = 4 * math.pi * model.a22 * model.a0 / l_rho

		self.g = {
			(COMP_1_minus1, COMP_1_minus1): self.g11,
			(COMP_1_minus1, COMP_2_1): self.g12,
			(COMP_2_1, COMP_1_minus1): self.g12,
			(COMP_2_1, COMP_2_1): self.g22
		}

		self.N = model.N

		self.xmax = model.border * math.sqrt(2.0 * self.muTF(comp=COMP_1_minus1))
		self.ymax = self.xmax
		self.zmax = self.xmax * self.lambda_

		# space step
		self.dx = 2.0 * self.xmax / (self.nvx - 1)
		self.dy = 2.0 * self.ymax / (self.nvy - 1)
		self.dz = 2.0 * self.zmax / (self.nvz - 1)
		self.dV = self.dx * self.dy * self.dz

		self.nvx_pow = log2(self.nvx)
		self.nvy_pow = log2(self.nvy)
		self.nvz_pow = log2(self.nvz)

		# k step
		self.dkx = math.pi / self.xmax
		self.dky = math.pi / self.ymax
		self.dkz = math.pi / self.zmax

		self.itmax = model.itmax
		self.dt_steady = model.dt_steady / self.t_rho
		self.dt_evo = model.dt_evo / self.t_rho
		self.ensembles = model.ensembles
		self.ens_shape = (self.ensembles * self.nvz, self.nvy, self.nvx)

		def recursiveCast(cast, obj):
			if isinstance(obj, dict):
				return dict([(key, recursiveCast(cast, obj[key])) for key in obj])
			elif isinstance(obj, list):
				return [recursiveCast(cast, elem) for elem in obj]
			elif isinstance(obj, float):
				return cast(obj)
			else:
				return obj

		self.__dict__ = recursiveCast(self.scalar.cast, self.__dict__)

	def muTF(self, comp=COMP_1_minus1, N=None):
		"""get TF-approximated chemical potential"""
		if N is None:
			N = self.N

		g = self.g[(comp, comp)]

		return self.scalar.cast((15.0 * N * g / (16.0 * math.pi * self.lambda_ * math.sqrt(2.0))) ** 0.4)
