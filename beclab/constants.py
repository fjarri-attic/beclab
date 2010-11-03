"""
Module, containing class with calculation constants
"""

import copy
import math
import numpy

from .globals import *
from .helpers import *

PSI_FUNC = 0
WIGNER = 1

COMP_1_minus1 = 0
COMP_2_1 = 1


class Constants:
	"""Calculation constants, in natural units"""

	def __init__(self, model, double=False):

		self.hbar = 1.054571628e-34 # Planck constant
		self.m = model.m
		a0 = 5.2917720859e-11 # Bohr radius

		model = copy.deepcopy(model)
		self.double = double

		precision = double_precision if double else single_precision
		self.scalar = precision.scalar
		self.complex = precision.complex

		self.ensembles = model.ensembles

		self.l111 = model.gamma111
		self.l12 = model.gamma12
		self.l22 = model.gamma22

		g11 = 4.0 * math.pi * (self.hbar ** 2) * model.a11 * a0 / self.m
		g12 = 4.0 * math.pi * (self.hbar ** 2) * model.a12 * a0 / self.m
		g22 = 4.0 * math.pi * (self.hbar ** 2) * model.a22 * a0 / self.m

		self.g = {
			(COMP_1_minus1, COMP_1_minus1): g11,
			(COMP_1_minus1, COMP_2_1): g12,
			(COMP_2_1, COMP_1_minus1): g12,
			(COMP_2_1, COMP_2_1): g22
		}

		self.N = model.N

		# trap frequencies
		self.w_x = 2.0 * math.pi * model.fx
		self.w_y = 2.0 * math.pi * model.fy
		self.w_z = 2.0 * math.pi * model.fz

		if model.nvx == 1 and model.nvy == 1:
			self.dim = 1
			self.nvz = model.nvz
			self.shape = (self.nvz,)
			self.ens_shape = (self.nvz * self.ensembles,)
			self.cells = self.nvz

			l_rho = math.sqrt(self.hbar / (2.0 * self.m * self.w_x))
			eff_area = 4.0 * math.pi * (l_rho ** 2)

			for key in self.g:
				self.g[key] /= eff_area

			self.l111 /= eff_area ** 2
			self.l12 /= eff_area
			self.l22 /= eff_area

		elif model.nvx == 1:
			raise Exception("2D clouds are not supported at the time")
		else:
			self.dim = 3
			self.nvx = model.nvx
			self.nvy = model.nvy
			self.nvz = model.nvz
			self.cells = self.nvx * self.nvy * self.nvz
			self.shape = (self.nvz, self.nvy, self.nvx)
			self.ens_shape = (self.nvz * self.ensembles, self.nvy, self.nvx)

		# prefix "w_" stands for radial frequency, "f_" for common frequency

		self.w_detuning = 2.0 * math.pi * model.detuning
		self.w_rabi = 2.0 * math.pi * model.rabi_freq
		self.t_rabi = 1.0 / model.rabi_freq

		# g itself is too small for single precision
		self.g_by_hbar = dict((key, self.g[key] / self.hbar) for key in self.g)

		mu1 = self.muTF(comp=COMP_1_minus1)

		self.e_cut = model.e_cut * mu1

		if self.dim > 2: self.xmax = model.border * math.sqrt(2.0 * mu1 / (self.m * self.w_x ** 2))
		if self.dim > 2: self.ymax = model.border * math.sqrt(2.0 * mu1 / (self.m * self.w_y ** 2))
		self.zmax = model.border * math.sqrt(2.0 * mu1 / (self.m * self.w_z ** 2))

		# space step
		if self.dim > 2: self.dx = 2.0 * self.xmax / (self.nvx - 1)
		if self.dim > 2: self.dy = 2.0 * self.ymax / (self.nvy - 1)
		self.dz = 2.0 * self.zmax / (self.nvz - 1)

		if self.dim == 1:
			self.dV = self.dz
		else:
			self.dV = self.dx * self.dy * self.dz

		self.V = self.dV * self.cells

		if self.dim > 2: self.nvx_pow = log2(self.nvx)
		if self.dim > 2: self.nvy_pow = log2(self.nvy)
		self.nvz_pow = log2(self.nvz)

		# k step
		if self.dim > 2: self.dkx = math.pi / self.xmax
		if self.dim > 2: self.dky = math.pi / self.ymax
		self.dkz = math.pi / self.zmax

		self.itmax = model.itmax
		self.dt_steady = model.dt_steady
		self.dt_evo = model.dt_evo

		# natural units
		self.t_rho = 1.0 / self.w_z
		self.e_rho = self.hbar * self.w_z
		self.l_rho = math.sqrt(self.hbar / (2.0 * self.m * self.w_z))

		#l_healing = 1.0 / math.sqrt(8.0 * math.pi * model.a11 * a0 * self.muTF() / g11)
		#print "nz >> " + str(self.zmax * 2.0 / l_healing)
		#print "nz << " + str(self.zmax * 2.0 / (model.a11 * a0))

		_, self.projector_modes = getProjectorArray(self)

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

		if self.dim == 3:
			return self._muTF3D(g, N)
		else:
			return self._muTF1D(g, N)

	def _muTF3D(self, g, N):
		w = (self.w_x * self.w_y * self.w_z) ** (1.0 / 3)
		return ((15 * N / (8.0 * math.pi)) ** 0.4) * \
			((self.m * w * w / 2) ** 0.6) * \
			(g ** 0.4)

	def _muTF1D(self, g, N):
		return ((0.75 * g * N) ** (2.0 / 3)) * \
			((self.m * self.w_z * self.w_z / 2) ** (1.0 / 3))
