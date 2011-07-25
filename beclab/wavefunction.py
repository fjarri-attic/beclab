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

			self._mdata = self._env.allocate(self._mshape, dtype)
			self.data = self._data = self._env.allocate(self._shape, dtype)

	def _gpu__prepare_specific(self):
		kernel_template = """
			<%!
				from math import sqrt
			%>

			EXPORTED_FUNC void fillWithZeros(int gsize, GLOBAL_MEM COMPLEX *res)
			{
				LIMITED_BY(gsize);
				%for comp in xrange(p.components):
					res[gsize * ${comp} + GLOBAL_INDEX] = complex_ctr(0, 0);
				%endfor
			}

			// Initialize ensembles with the copy of the current state
			EXPORTED_FUNC void fillEnsembles(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *src)
			{
				LIMITED_BY(gsize);
				%for comp in xrange(p.components):
					res[GLOBAL_INDEX + gsize * ${comp}] =
						src[GLOBAL_INDEX % (gsize / ${p.ensembles}) + gsize * ${comp}];
				%endfor
			}

			EXPORTED_FUNC void addVacuumParticles(int gsize, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM COMPLEX *randoms, GLOBAL_MEM SCALAR *mask)
			{
				LIMITED_BY(gsize);

				SCALAR mask_elem = mask[GLOBAL_INDEX % (gsize / ${p.ensembles})];
				COMPLEX val;
				int id;

				%for comp in xrange(p.components):
					id = GLOBAL_INDEX + gsize * ${comp};
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
