"""
Different meters for particle states (measuring particles number, energy and so on)
"""

import numpy

from .helpers import *
from .constants import *


class Wavefunction(PairedCalculation):

	def __init__(self, env, constants, grid, comp=0, prepare=True):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self.type = CLASSICAL
		self.comp = comp
		self.in_mspace = False

		if prepare:
			if isinstance(grid, UniformGrid):
				self._plan = createFFTPlan(env, constants, grid)
			elif isinstance(grid, HarmonicGrid):
				self._plan = createFHTPlan(env, constants, grid, 1)

			self._random = createRandom(env, constants.double)
			self._prepare()
			self._initializeMemory()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernel_template = """
			<%!
				from math import sqrt
			%>

			EXPORTED_FUNC void fillWithZeros(GLOBAL_MEM COMPLEX *res)
			{
				DEFINE_INDEXES;
				res[index] = complex_ctr(0, 0);
			}

			// Initialize ensembles with the copy of the current state
			EXPORTED_FUNC void fillEnsembles(GLOBAL_MEM COMPLEX *src,
				GLOBAL_MEM COMPLEX *dest, int ensembles)
			{
				DEFINE_INDEXES;
				COMPLEX src_val = src[index];

				for(int i = 0; i < ensembles; i++)
					dest[index + i * ${g.size}] = src_val;
			}

			EXPORTED_FUNC void addVacuumParticles(GLOBAL_MEM COMPLEX *modespace_data,
				GLOBAL_MEM COMPLEX *randoms, GLOBAL_MEM SCALAR *mask)
			{
				DEFINE_INDEXES;
				SCALAR mask_elem = mask[cell_index];
				COMPLEX val = modespace_data[index];

				val = val + randoms[index]; // add noise
				val = complex_mul_scalar(val, mask_elem); // remove high-energy components

				modespace_data[index] = val;
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants, self._grid)
		self._kernel_fillWithZeros = self._program.fillWithZeros
		self._kernel_fillEnsembles = self._program.fillEnsembles
		self._kernel_addVacuumParticles = self._program.addVacuumParticles

	def _cpu__kernel_addVacuumParticles(self, _, modespace_data, randoms, mask):
		tile = (self.shape[0],) + (1,) * (self._grid.dim - 1)
		modespace_data += randoms # add vacuum particles
		modespace_data *= numpy.tile(mask, tile) # remove high-energy components

	def _cpu__kernel_fillWithZeros(self, _, data):
		data.flat[:] = numpy.zeros_like(data).flat

	def _cpu__kernel_fillEnsembles(self, _, data, new_data, ensembles):
		tile = (ensembles,) + (1,) * self._grid.dim
		new_data.flat[:] = numpy.tile(data, tile).flat

	def _initializeMemory(self, ensembles=None, data=None, mspace=None):
		dtype = self._constants.complex.dtype

		if mspace is not None:
			self.in_mspace = mspace

		if ensembles is None:
			ensembles = 1 if data is None else data.shape[0]

		if isinstance(self._grid, UniformGrid):
			self.shape = self._shape = self._mshape = (ensembles,) + self._grid.shape
			self.size = self._size = self._msize = self._grid.size
			self.data = self._data = self._mdata = self._env.allocate(self.shape, dtype)

		else:
			self._shape = (ensembles,) + self._grid.shape
			self._mshape = (ensembles,) + self._grid.mshape
			self.shape = self._mshape if self.in_mspace else self._shape

			self._size = self._grid.size
			self._msize = self._grid.msize
			self.size = self._msize if self.in_mspace else self._size

			self._mdata = self._env.allocate(self._mshape, dtype)
			self._data = self._env.allocate(self._shape, dtype)
			self.data = self._data

		if data is None:
			self._kernel_fillWithZeros(self.size, self.data)
		elif data.shape[0] == ensembles:
			assert data.shape == self.shape
			self._env.copyBuffer(data, dest=self.data)
		elif data.shape[0] == 1 and ensembles != 1:
			assert data.shape[1:] == self.shape[1:]
			self._kernel_fillEnsembles(self.size, data, self.data, numpy.int32(ensembles))

	def toMSpace(self):
		assert not self.in_mspace
		self._plan.execute(self._data, self._mdata, batch=self.shape[0])
		self.size = self._msize
		self.shape = self._mshape
		self.data = self._mdata
		self.in_mspace = True

	def toXSpace(self):
		assert self.in_mspace
		self._plan.execute(self._mdata, self._data, batch=self.shape[0], inverse=True)
		self.size = self._size
		self.shape = self._shape
		self.data = self._data
		self.in_mspace = False

	def _addVacuumParticles(self, randoms, mask):

		dtype = self.data.dtype
		ensembles = self.shape[0]

		was_in_xspace = not self.in_mspace

		if was_in_xspace:
			self.toMSpace()

		self._kernel_addVacuumParticles(self.size, self.data, randoms, mask)

		if was_in_xspace:
			self.toXSpace()

	def createEnsembles(self, ensembles):
		assert self.shape[0] == 1
		self._initializeMemory(ensembles=ensembles, data=self.data)

	def toWigner(self, ensembles):

		assert self.type == CLASSICAL

		self.createEnsembles(ensembles)

		# FIXME: in fact, we only need randoms in cells where projector mask == 1
		# Scaling assumes that in modespace wavefunction normalized on atom number
		randoms = self._random.random_normal(self.shape, scale=numpy.sqrt(0.5))
		projector_mask = getProjectorMask(self._env, self._constants, self._grid)
		self._addVacuumParticles(randoms, projector_mask)

		self.type = WIGNER

	def copy(self, prepare=True):
		res = Wavefunction(self._env, self._constants, self._grid, comp=self.comp, prepare=prepare)
		res._initializeMemory(data=self.data, mspace=self.in_mspace)
		res.type = self.type
		return res


class TwoComponentCloud:

	def __init__(self, env, constants, grid, psi0=None, psi1=None, prepare=True):
		assert psi0 is not None or psi1 is not None
		assert psi0 is None or psi1 is None or psi0.type == psi1.type
		assert psi0 is None or psi0.comp == 0
		assert psi1 is None or psi1.comp == 1

		self._env = env
		self._constants = constants.copy()
		self._grid = grid.copy()

		self.type = psi1.type if psi0 is None else psi0.type
		self.time = 0.0

		self.psi0 = psi0.copy(prepare=prepare) \
			if psi0 is not None else Wavefunction(env, constants, grid, type=self.type, comp=0)
		self.psi1 = psi1.copy(prepare=prepare) \
			if psi1 is not None else Wavefunction(env, constants, grid, type=self.type, comp=1)

	def toWigner(self, ensembles):
		self.psi0.toWigner(ensembles)
		self.psi1.toWigner(ensembles)
		self.type = self.psi0.type

	def copy(self, prepare=True):
		res = TwoComponentCloud(self._env, self._constants, self._grid,
			psi0=self.psi0, psi1=self.psi1, prepare=prepare)
		res.time = self.time
		return res

	def createEnsembles(self, ensembles):
		self.psi0.createEnsembles(ensembles)
		self.psi1.createEnsembles(ensembles)
