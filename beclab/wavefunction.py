"""
Different meters for particle states (measuring particles number, energy and so on)
"""

import numpy

from .helpers import *
from .constants import *


class WavefunctionSet(PairedCalculation):

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self.type = CLASSICAL
		self.in_mspace = False

		if isinstance(grid, UniformGrid):
			self._plan = createFFTPlan(env, constants, grid)
		elif isinstance(grid, HarmonicGrid):
			self._plan = createFHTPlan(env, constants, grid, 1)

		self._random = createRandom(env, constants.double)

		self._addParameters(components=2, ensembles=1)
		self.prepare(**kwds)

		self._kernel_fillWithZeros(self.size, self.data)

	def _prepare(self):
		dtype = self._constants.complex.dtype

		self.ensembles = self._p.ensembles
		self.components = self._p.components

		if isinstance(self._grid, UniformGrid):
			self.shape = self._shape = self._mshape = \
				(self.components, self.ensembles,) + self._grid.shape
			self.size = self._size = self._msize = self._grid.size * self.ensembles
			self.data = self._data = self._mdata = self._env.allocate(self.shape, dtype)

		else:
			self.shape = self._shape = (self.components, self.ensembles,) + self._grid.shape
			self._mshape = (self.components, self.ensembles,) + self._grid.mshape

			self.size = self._size = self._grid.size * self.ensembles
			self._msize = self._grid.msize

			self.data = self._mdata = self._env.allocate(self._mshape, dtype)
			self._data = self._env.allocate(self._shape, dtype)

	def _gpu__prepare_specific(self):
		kernel_template = """
			<%!
				from math import sqrt
			%>

			EXPORTED_FUNC void fillWithZeros(GLOBAL_MEM COMPLEX *res)
			{
				LIMITED_BY_GRID;
				%for component in xrange(p.components):
					res[${component * g.size} + GLOBAL_INDEX] = complex_ctr(0, 0);
				%endfor
			}

			// Initialize ensembles with the copy of the current state
			EXPORTED_FUNC void fillEnsembles(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *src)
			{
				LIMITED_BY(${p.ensembles});
				%for component in xrange(p.components):
					res[GLOBAL_INDEX + ${component * g.size * p.ensembles}] =
						src[CELL_INDEX + ${component * g.size}];
				%endfor
			}

			EXPORTED_FUNC void addVacuumParticles(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *randoms, GLOBAL_MEM SCALAR *mask)
			{
				LIMITED_BY(${p.ensembles});

				SCALAR mask_elem = mask[CELL_INDEX];
				COMPLEX val;
				int id;

				%for component in xrange(p.components):
					id = GLOBAL_INDEX + ${component * g.size * p.ensembles};
					val = data[id];
					val = val + randoms[id]; // add noise
					val = complex_mul_scalar(val, mask_elem);
					data[id] = val;
				%endfor
			}
		"""

		self.__program = self.compileProgram(kernel_template)
		self._kernel_fillWithZeros = self.__program.fillWithZeros
		self._kernel_fillEnsembles = self.__program.fillEnsembles
		self._kernel_addVacuumParticles = self.__program.addVacuumParticles

	def _cpu__kernel_addVacuumParticles(self, gsize, modespace_data, randoms, mask):
		tile = (self.components, self.ensembles,) + (1,) * self._grid.dim
		modespace_data += randoms # add vacuum particles
		modespace_data *= numpy.tile(mask, tile) # remove high-energy components

	def _cpu__kernel_fillWithZeros(self, gsize, data):
		data.flat[:] = numpy.zeros_like(data).flat

	def _cpu__kernel_fillEnsembles(self, gsize, data, new_data):
		tile = (self.components, self.ensembles,) + (1,) * self._grid.dim
		new_data.flat[:] = numpy.tile(data, tile).flat

	def toMSpace(self):
		assert not self.in_mspace
		self._plan.execute(self._data, self._mdata,
			batch=self.ensembles * self.components)
		self.size = self._msize
		self.shape = self._mshape
		self.data = self._mdata
		self.in_mspace = True

	def toXSpace(self):
		assert self.in_mspace
		self._plan.execute(self._mdata, self._data,
			batch=self.ensembles * self.components, inverse=True)
		self.size = self._size
		self.shape = self._shape
		self.data = self._data
		self.in_mspace = False

	def _addVacuumParticles(self, randoms, mask):

		dtype = self.data.dtype
		was_in_xspace = not self.in_mspace

		if was_in_xspace:
			self.toMSpace()

		self._kernel_addVacuumParticles(self.size, self.data, randoms, mask)

		if was_in_xspace:
			self.toXSpace()

	def createEnsembles(self, ensembles):
		assert self.ensembles == 1
		data = self.data
		self.prepare(ensembles=ensembles)
		self._kernel_fillEnsembles(self.size, self.data, data)

	def toWigner(self, ensembles):

		assert self.type == CLASSICAL

		self.createEnsembles(ensembles)

		# FIXME: in fact, we only need randoms in cells where projector mask == 1
		# Scaling assumes that in modespace wavefunction normalized on atom number
		randoms = self._random.random_normal(self.shape, scale=numpy.sqrt(0.5))
		projector_mask = getProjectorMask(self._env, self._constants, self._grid)
		self._addVacuumParticles(randoms, projector_mask)

		self.type = WIGNER

	def copy(self):
		res = WavefunctionSet(self._env, self._constants, self._grid,
			components=self.components, ensembles=self.ensembles)
		if self.in_mspace:
			res.toMSpace()
		self._env.copyBuffer(self.data, dest=res.data)
		res.type = self.type
		return res

	def fillComponent(self, target_comp, psi, source_comp):
		assert self._grid == psi._grid and self.ensembles == psi.ensembles
		comp_size = self.ensembles * self._grid.size
		self._env.copyBuffer(psi.data, self.data,
			src_offset=source_comp * comp_size,
			dest_offset=target_comp * comp_size,
			length=comp_size)


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
				LIMITED_BY_GRID;
				res[GLOBAL_INDEX] = complex_ctr(0, 0);
			}

			// Initialize ensembles with the copy of the current state
			EXPORTED_FUNC void fillEnsembles(GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *src, int ensembles)
			{
				LIMITED_BY(ensembles);
				COMPLEX val = src[CELL_INDEX];
				res[GLOBAL_INDEX] = val;
			}

			EXPORTED_FUNC void addVacuumParticles(GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *randoms, GLOBAL_MEM SCALAR *mask, int ensembles)
			{
				LIMITED_BY(ensembles);

				SCALAR mask_elem = mask[CELL_INDEX];
				COMPLEX val = data[GLOBAL_INDEX];

				val = val + randoms[GLOBAL_INDEX]; // add noise
				val = complex_mul_scalar(val, mask_elem); // remove high-energy components

				data[GLOBAL_INDEX] = val;
			}
		"""

		self._program = self.compileProgram(kernel_template)
		self._kernel_fillWithZeros = self._program.fillWithZeros
		self._kernel_fillEnsembles = self._program.fillEnsembles
		self._kernel_addVacuumParticles = self._program.addVacuumParticles

	def _cpu__kernel_addVacuumParticles(self, gsize, modespace_data, randoms, mask, ensembles):
		tile = (ensembles,) + (1,) * self._grid.dim
		modespace_data += randoms # add vacuum particles
		modespace_data *= numpy.tile(mask, tile) # remove high-energy components

	def _cpu__kernel_fillWithZeros(self, gsize, data):
		data.flat[:] = numpy.zeros_like(data).flat

	def _cpu__kernel_fillEnsembles(self, gsize, data, new_data, ensembles):
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
		# If nothing is given, initialize with empty wavefunction
		if psi0 is None and psi1 is None:
			psi0 = Wavefunction(env, constants, grid, comp=0)

		assert psi0 is None or psi0.comp == 0
		assert psi1 is None or psi1.comp == 1

		self._env = env
		self._constants = constants.copy()
		self._grid = grid.copy()

		self.time = 0.0

		self.psi0 = psi0.copy(prepare=prepare) \
			if psi0 is not None else Wavefunction(env, constants, grid, comp=0)
		self.psi1 = psi1.copy(prepare=prepare) \
			if psi1 is not None else Wavefunction(env, constants, grid, comp=1)

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
