import numpy

from .helpers import *
from .constants import *


class DensityMeter(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self._dV = grid.dV_device

		self._sreduce_cex_to_cx = createReduce(env, constants.scalar.dtype)
		self._sreduce_cex_to_c = createReduce(env, constants.scalar.dtype)

		self._sreduce_cem_to_cm = createReduce(env, constants.scalar.dtype)
		self._sreduce_cem_to_c = createReduce(env, constants.scalar.dtype)

		self._sreduce_cem_to_ce = createReduce(env, constants.scalar.dtype)
		self._sreduce_cex_to_ce = createReduce(env, constants.scalar.dtype)

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)
		self.prepare(**kwds)

	def _prepare(self):

		if self._p.psi_type == REPR_WIGNER:
			self._p.modify_density = True
			self._density_modifiers = self._grid.density_modifiers_device
		else:
			self._p.modify_density = False

			# This is supposed to be a NULL pointer.
			# No idea how to do it properly in PyCuda.
			self._density_modifiers = self._env.allocate((1,), self._constants.scalar.dtype)

		scalar_t = self._constants.scalar.dtype
		comp = self._p.components
		ens = self._p.ensembles
		xsize = self._grid.size
		msize = self._grid.msize
		xshape = self._grid.shape
		mshape = self._grid.mshape

		self._sreduce_cex_to_cx.prepare(sparse=True,
			batch=comp, length=ens * xsize, final_length=xsize)
		self._sreduce_cex_to_c.prepare(sparse=False,
			length=comp * ens * xsize, final_length=comp)
		self._sreduce_cem_to_cm.prepare(sparse=True,
			batch=comp, length=ens * msize, final_length=msize)
		self._sreduce_cem_to_c.prepare(sparse=False,
			length=comp * ens * msize, final_length=comp)
		self._sreduce_cex_to_ce.prepare(sparse=False,
			length=comp * ens * xsize, final_length=comp * ens)
		self._sreduce_cem_to_ce.prepare(sparse=False,
			length=comp * ens * msize, final_length=comp * ens)

		# Methods of this class can be called either for mode-space or for x-space psi
		# In either case, the buffers for remaining space will be unused,
		# so we can safely use views
		max_size = max(xsize, msize)

		main_buffer = self._env.allocate((comp, ens, max_size), scalar_t)

		self._sbuffer_cex = getView(main_buffer, (comp, ens) + xshape)
		self._sbuffer_cem = getView(main_buffer, (comp, ens) + mshape)

		self._sbuffer_c = self._env.allocate((comp,), scalar_t)
		self._sbuffer_ce = self._env.allocate((comp, ens), scalar_t)

		# Warning: used as a result of _sreduce_cex_to_cx(),
		# which is sparse and therefore view-safe
		self._sbuffer_cx = getView(main_buffer, (comp, 1) + xshape)
		self._sbuffer_cm = getView(main_buffer, (comp, 1) + mshape)

	def _gpu__prepare_specific(self):
		kernel_template = """
			%for space in ('m', 'x'):
			EXPORTED_FUNC void ${space}density(int gsize, GLOBAL_MEM SCALAR *res,
				GLOBAL_MEM COMPLEX *state, GLOBAL_MEM SCALAR *modifiers, int coeff)
			{
				LIMITED_BY(gsize);
				int id;
				SCALAR modifier;

				%if p.modify_density:
				modifier = modifiers[GLOBAL_INDEX % (gsize / ${p.ensembles})];
				%if space == 'm': ## using modifiers array as a mask only
				modifier = (modifier == 0 ? 0 : 0.5);
				%endif
				%else:
				modifier = 0;
				%endif

				%for comp in xrange(p.components):
				id = GLOBAL_INDEX + gsize * ${comp};
				res[id] = (squared_abs(state[id]) - modifier) / coeff;
				%endfor
			}
			%endfor

			EXPORTED_FUNC void multiplyTiledSS(int gsize,
				GLOBAL_MEM SCALAR *data, GLOBAL_MEM SCALAR *coeffs, int ensembles)
			{
				LIMITED_BY(gsize);

				SCALAR coeff_val = coeffs[GLOBAL_INDEX % (gsize / ensembles)];
				SCALAR data_val;

				%for comp in xrange(p.components):
				data_val = data[GLOBAL_INDEX + gsize * ${comp}];
				data[GLOBAL_INDEX + gsize * ${comp}] = data_val * coeff_val;
				%endfor
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_xdensity = self._program.xdensity
		self._kernel_mdensity = self._program.mdensity
		self._kernel_multiplyTiledSS = self._program.multiplyTiledSS

	def _cpu__kernel_density(self, gsize, density, data, modifiers, coeff, in_mspace):
		n = numpy.abs(data) ** 2

		if self._p.modify_density:
			if in_mspace:
				modifiers = (modifiers > 0).astype(self._constants.scalar.dtype) * 0.5
			modifiers = numpy.tile(modifiers,
				(self._p.components, self._p.ensembles,) + (1,) * self._grid.dim)
		else:
			modifiers = numpy.zeros_like(n)

		self._env.copyBuffer((n - modifiers) / coeff, dest=density)

	def _cpu__kernel_xdensity(self, gsize, density, data, modifiers, coeff):
		self._kernel_density(gsize, density, data, modifiers, coeff, False)

	def _cpu__kernel_mdensity(self, gsize, density, data, modifiers, coeff):
		self._kernel_density(gsize, density, data, modifiers, coeff, True)

	def _cpu__kernel_multiplyTiledSS(self, gsize, data, coeffs, ensembles):
		data *= numpy.tile(coeffs,
			(self._p.components, ensembles,) + (1,) * self._grid.dim)

	def getNDensity(self, psi, coeff=1):
		"""Returns population density for every point on a lattice"""

		if psi.in_mspace:
			self._kernel_mdensity(psi.size, self._sbuffer_cem, psi.data,
				self._density_modifiers, numpy.int32(coeff))
			return self._sbuffer_cem
		else:
			self._kernel_xdensity(psi.size, self._sbuffer_cex, psi.data,
				self._density_modifiers, numpy.int32(coeff))
			return self._sbuffer_cex

	def getN(self, psi, coeff=1):
		"""Returns population for every point on a lattice"""

		density = self.getNDensity(psi, coeff=coeff)
		if not psi.in_mspace:
			self._kernel_multiplyTiledSS(psi.size, density, self._dV,
				numpy.int32(self._p.ensembles))
		return density

	def getNDensityAverage(self, psi):
		"""Returns population density averaged over ensembles"""

		ensembles = psi.ensembles
		components = psi.components
		size = self._grid.msize if psi.in_mspace else self._grid.size

		density = self.getNDensity(psi, coeff=ensembles)
		if psi.in_mspace:
			self._sreduce_cem_to_cm(density, self._sbuffer_cm)
			return self._sbuffer_cm
		else:
			self._sreduce_cex_to_cx(density, self._sbuffer_cx)
			return self._sbuffer_cx

	def getNAverage(self, psi):
		"""Returns population averaged over ensembles"""

		density = self.getNDensityAverage(psi)
		if not psi.in_mspace:
			self._kernel_multiplyTiledSS(self._grid.size, density, self._dV, numpy.int32(1))
		return density

	def getNTotal(self, psi):
		"""Returns total population for each component"""

		p = self.getN(psi, coeff=self._p.ensembles)
		if psi.in_mspace:
			self._sreduce_cem_to_c(p, self._sbuffer_c)
		else:
			self._sreduce_cex_to_c(p, self._sbuffer_c)
		Ns = self._env.fromDevice(self._sbuffer_c)
		return Ns

	def getNPerEnsemble(self, psi):
		"""Returns total population for each component in each ensemble"""

		p = self.getN(psi, coeff=self._p.ensembles)
		if psi.in_mspace:
			self._sreduce_cem_to_ce(p, self._sbuffer_ce)
		else:
			self._sreduce_cex_to_ce(p, self._sbuffer_ce)
		return self._sbuffer_ce


class InteractionMeter(PairedCalculation):

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self._potentials = grid.potentials_device
		self._energy = grid.energy_device
		self._dV = grid.dV_device

		self._creduce_cex_to_c = createReduce(env, constants.complex.dtype)
		self._creduce_ex_to_e = createReduce(env, constants.complex.dtype)
		self._creduce_ex_to_1 = createReduce(env, constants.complex.dtype)

		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)
		self.prepare(**kwds)

	def _prepare(self):
		self._p.g = self._constants.g / self._constants.hbar
		self._p.need_potentials = isinstance(self._grid, UniformGrid)

		if self._p.psi_type == REPR_WIGNER:
			self._density_modifiers = self._env.toDevice(self._grid.density_modifiers)
		else:
			self._density_modifiers = self._env.toDevice(
				numpy.zeros(self._grid.shape).astype(self._constants.scalar.dtype))

		scalar_t = self._constants.scalar.dtype
		complex_t = self._constants.complex.dtype
		comp = self._p.components
		ens = self._p.ensembles
		xsize = self._grid.size
		msize = self._grid.msize
		xshape = self._grid.shape
		mshape = self._grid.mshape

		self._creduce_cex_to_c.prepare(sparse=False, length=comp * ens * xsize, final_length=comp)
		self._creduce_ex_to_1.prepare(sparse=False, length=ens * xsize, final_length=1)
		self._creduce_ex_to_e.prepare(sparse=False,	length=ens * xsize, final_length=ens)

		self._cbuffer_cex = self._env.allocate((comp, ens) + xshape, complex_t)
		self._cbuffer_cem = self._env.allocate((comp, ens) + mshape, complex_t)
		self._cbuffer_ex = getView(self._cbuffer_cex, (1, ens) + xshape)

		self._cbuffer_1 = self._env.allocate((1,), complex_t)
		self._cbuffer_e = self._env.allocate((ens,), complex_t)
		self._cbuffer_c = self._env.allocate((comp,), complex_t)

	def _gpu__prepare_specific(self):
		kernel_template = """
			EXPORTED_FUNC void interaction(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *data)
			{
				LIMITED_BY(gsize);
				COMPLEX val0 = data[GLOBAL_INDEX + gsize * ${0}];
				COMPLEX val1 = data[GLOBAL_INDEX + gsize * ${1}];

				res[GLOBAL_INDEX] = complex_mul(val0, conj(val1));
			}

			EXPORTED_FUNC void invariant(int gsize, GLOBAL_MEM COMPLEX *res_mdata,
				GLOBAL_MEM COMPLEX *xdata, GLOBAL_MEM SCALAR *potentials, int coeff)
			{
				LIMITED_BY(gsize);

				%if p.need_potentials:
				SCALAR potential = potentials[GLOBAL_INDEX % (gsize / ${p.ensembles})];
				%endif

				%for comp in xrange(p.components):
				int id${comp} = GLOBAL_INDEX + gsize * ${comp};
				COMPLEX xdata${comp} = xdata[id${comp}];
				COMPLEX mdata${comp} = res_mdata[id${comp}];
				SCALAR n${comp} = squared_abs(xdata${comp});
				%endfor

				%for comp in xrange(p.components):
				SCALAR nonlinear${comp} = ${'potential' if p.need_potentials else '0'};
					%for comp_other in xrange(p.components):
					nonlinear${comp} += (SCALAR)${p.g[comp, comp_other]} * n${comp_other} / coeff;
					%endfor
				nonlinear${comp} *= n${comp};
				COMPLEX differential${comp} = complex_mul(conj(xdata${comp}), mdata${comp});

				// integral over imaginary part is zero
				res_mdata[id${comp}] = complex_ctr(nonlinear${comp}, 0) + differential${comp};
				%endfor
			}

			%if c.so_coupling:
			EXPORTED_FUNC void multiplySOEnergy(int gsize,
				GLOBAL_MEM COMPLEX *mdata, GLOBAL_MEM COMPLEX *energy)
			{
				LIMITED_BY(gsize);

				COMPLEX energy00 = energy[GLOBAL_INDEX];
				COMPLEX energy01 = energy[GLOBAL_INDEX + ${g.msize}];
				COMPLEX energy10 = energy[GLOBAL_INDEX + ${g.msize * 2}];
				COMPLEX energy11 = energy[GLOBAL_INDEX + ${g.msize * 3}];

				COMPLEX data0 = mdata[GLOBAL_INDEX];
				COMPLEX data1 = mdata[GLOBAL_INDEX + ${g.msize}];

				mdata[GLOBAL_INDEX] = complex_mul(data0, energy00) + complex_mul(data1, energy01);
				mdata[GLOBAL_INDEX + ${g.msize}] = complex_mul(data0, energy10) +
					complex_mul(data1, energy11);
			}
			%endif

			EXPORTED_FUNC void multiplyTiledCS(int gsize,
				GLOBAL_MEM COMPLEX *data, GLOBAL_MEM SCALAR *coeffs, int components)
			{
				LIMITED_BY(gsize);

				SCALAR coeff_val = coeffs[GLOBAL_INDEX % (gsize / ${p.ensembles})];
				COMPLEX data_val;

				for(int comp = 0; comp < components; comp++)
				{
					data_val = data[GLOBAL_INDEX + gsize * comp];
					data[GLOBAL_INDEX + gsize * comp] =
						complex_mul_scalar(data_val, coeff_val);
				}
			}
		"""

		self._program = self.compileProgram(kernel_template)

		self._kernel_interaction = self._program.interaction
		self._kernel_invariant = self._program.invariant
		self._kernel_multiplyTiledCS = self._program.multiplyTiledCS

		if self._constants.so_coupling:
			self._kernel_multiplySOEnergy = self._program.multiplySOEnergy

	def _cpu__kernel_interaction(self, gsize, res, data):
		self._env.copyBuffer(data[0] * data[1].conj(), dest=res)

	def _cpu__kernel_multiplyTiledCS(self, gsize, data, coeffs, components):
		data *= numpy.tile(coeffs,
			(components, self._p.ensembles,) + (1,) * self._grid.dim)

	def _cpu__kernel_invariant(self, gsize, res_mdata, xdata, potentials, coeff):

		tile = (self._p.ensembles,) + (1,) * self._grid.dim
		g = self._p.g
		n = numpy.abs(xdata) ** 2
		components = self._p.components

		res_mdata *= xdata.conj()

		if self._p.need_potentials:
			for comp in xrange(components):
				res_mdata[comp] += numpy.tile(potentials, tile) * n[comp]

		for comp in xrange(components):
			for comp_other in xrange(components):
				res_mdata[comp] += n[comp] * (g[comp, comp_other] * n[comp_other] / coeff)

	def _cpu__kernel_multiplySOEnergy(self, msize, mdata, energy):
		mdata_copy = mdata.copy()
		mdata[0, 0] = mdata_copy[0, 0] * energy[0, 0] + mdata_copy[1, 0] * energy[0, 1]
		mdata[1, 0] = mdata_copy[0, 0] * energy[1, 0] + mdata_copy[1, 0] * energy[1, 1]

	def _getInvariant(self, psi, coeff):

		# TODO: work out the correct formula for Wigner function's E/mu
		if psi.type != REPR_CLASSICAL:
			raise NotImplementedError()

		so = self._constants.so_coupling

		batch = self._p.ensembles * self._p.components
		xsize = self._grid.size * self._p.ensembles
		msize = self._grid.msize * self._p.ensembles
		cast = self._constants.scalar.cast

		# FIXME: not a good way to provide transformation
		psi._plan.execute(psi.data, self._cbuffer_cem, batch=batch)
		if so:
			self._kernel_multiplySOEnergy(msize, self._cbuffer_cem, self._energy)
		else:
			self._kernel_multiplyTiledCS(msize, self._cbuffer_cem, self._energy,
				numpy.int32(self._p.components))
		psi._plan.execute(self._cbuffer_cem, self._cbuffer_cex,
			batch=batch, inverse=True)

		self._kernel_invariant(xsize, self._cbuffer_cex,
			psi.data, self._potentials, numpy.int32(coeff))
		self._kernel_multiplyTiledCS(xsize, self._cbuffer_cex, self._dV,
			numpy.int32(self._p.components))

		self._creduce_cex_to_c(self._cbuffer_cex, self._cbuffer_c)

		# We were doing operations on complex temporary array to save memory.
		# Now when we back on CPU we can safely discard imaginary part.
		comps = self._env.fromDevice(self._cbuffer_c).real

		return comps / self._p.ensembles * self._constants.hbar

	def getI(self, psi):
		"""Returns I = (Psi1^*) * Psi2 for each point of a lattice"""

		self._kernel_interaction(psi.size, self._cbuffer_ex, psi.data)
		self._kernel_multiplyTiledCS(psi.size, self._cbuffer_ex,
			self._dV, numpy.int32(1))
		return self._cbuffer_ex

	def getIPerEnsemble(self, psi):
		i = self.getI(psi)
		self._creduce_ex_to_e(i, self._cbuffer_e)
		return self._cbuffer_e

	def getITotal(self, psi):
		i = self.getI(psi)
		self._creduce_ex_to_1(i, self._cbuffer_1)
		return self._env.fromDevice(self._cbuffer_1)[0]

	def getETotal(self, psi):
		"""Returns total energy"""
		return self._getInvariant(psi, 2)

	def getMuTotal(self, psi):
		"""Returns total chemical potential"""
		return self._getInvariant(psi, 1)


class IntegralMeter:

	def __init__(self, env, constants, grid):
		pass

	def prepare(self, **kwds):
		pass

	@classmethod
	def forPsi(cls, psi):
		return cls(psi._env, psi._constants, psi._grid)

	def getVisibility(self, psi):
		assert psi.components == 2
		N = psi.density_meter.getNTotal()
		I = psi.interaction_meter.getITotal()
		return 2.0 * numpy.abs(I) / N.sum() / psi.ensembles

	def getEPerParticle(self, psi, N=None):
		if N is None:
			N = psi.density_meter.getNTotal().sum()
		return psi.interaction_meter.getETotal() / N

	def getMuPerParticle(self, psi, N=None):
		if N is None:
			N = psi.density_meter.getNTotal().sum()
		return psi.interaction_meter.getMuTotal() / N


class ProjectionMeter(PairedCalculation):

	def __init__(self, env, constants, grid, **kwds):
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid
		self._addParameters(components=2, ensembles=1, psi_type=REPR_CLASSICAL)
		self._reduce = createReduce(env, constants.scalar.dtype)
		self.prepare(**kwds)

	@classmethod
	def forPsi(cls, psi):
		return cls(psi._env, psi._constants, psi._grid,
			components=psi.components, ensembles=psi.ensembles, psi_type=psi.type)

	def _prepare(self):
		self._reduce.prepare(length=self._p.components * self._grid.size,
			final_length=self._grid.shape[0] * self._p.components)
		self._z_buffer = self._env.allocate(
			(self._p.components, self._grid.shape[0]), self._constants.scalar.dtype)

	def getXY(self, psi):
		# TODO: use reduction on device if it starts to take too much time
		p = self._env.fromDevice(psi.density_meter.getNAverage())
		nx = self._grid.shape[2]
		ny = self._grid.shape[1]

		# sum over ensembles (since it is only 1 of them, just removes this dimension)
		# and over z-axis
		xy = p.sum(1).sum(1)
		dy = self._grid.dy.reshape(self._grid.shape[1], 1)
		xy /= numpy.tile(dy, (self._p.components, 1, nx))
		xy /= numpy.tile(self._grid.dx, (self._p.components, ny, 1))
		return xy

	def getYZ(self, psi):
		# TODO: use reduction on device if it starts to take too much time
		p = self._env.fromDevice(psi.density_meter.getNAverage())
		ny = self._grid.shape[1]
		nz = self._grid.shape[0]

		# sum over ensembles (since it is only 1 of them, just removes this dimension)
		# and over x-axis
		yz = p.sum(1).sum(3)
		dz = self._grid.dz.reshape(self._grid.shape[0], 1)
		yz /= numpy.tile(dz, (self._p.components, 1, ny))
		yz /= numpy.tile(self._grid.dy, (self._p.components, nz, 1))
		return yz

	def getXYSlice(self, psi, z_index=None):
		# TODO: use reduction on device if it starts to take too much time
		p = self._env.fromDevice(psi.density_meter.getNDensityAverage())
		if z_index is None:
			z_index = self._grid.shape[0] / 2
		return p[:,0,z_index,:,:]

	def getYZSlice(self, psi, x_index=None):
		# TODO: use reduction on device if it starts to take too much time
		p = self._env.fromDevice(psi.density_meter.getNDensityAverage())
		if x_index is None:
			x_index = self._grid.shape[2] / 2
		return p[:,0,:,:,x_index]

	def getZ(self, psi):
		p = psi.density_meter.getNAverage()
		self._reduce(p, self._z_buffer)
		res = self._env.fromDevice(self._z_buffer)
		return res / numpy.tile(self._grid.dz, (self._p.components, 1))


class UncertaintyMeter:

	def __init__(self, env, constants, grid):
		self._env = env

	def prepare(self, **kwds):
		pass

	@classmethod
	def forPsi(cls, psi):
		return cls(psi._env, psi._constants, psi._grid,)

	def getPhaseNoise(self, psi):
		"""
		Warning: this function considers spin distribution ellipse to be horizontal,
		which is not always so.
		"""
		assert psi.components == 2

		# Complex numbers {S_xj + iS_yj, j = 1..N}
		i = self._env.fromDevice(psi.interaction_meter.getIPerEnsemble())

		phi = numpy.angle(i) # normalizing

		# Center of the distribution can be shifted to pi or -pi,
		# making mean() return incorrect values.
		# The following approximate method will allow us to shift the center to zero
		# It will work only if the maximum of the distribution is clearly
		# distinguished; otherwise it can give anything as a result

		Pperp = numpy.exp(1j * phi) # transforming Pperp to distribution on the unit circle
		Pmean = Pperp.mean() # Center of masses is supposed to be close to the center of distribution

		# Normalizing the direction to the center of masses
		# Now angle(Pmean) ~ proper mean of Pperp
		Pmean /= numpy.abs(Pmean)

		# Shifting the distribution
		Pcent = Pperp * Pmean.conj()
		phi_centered = numpy.angle(Pcent)

		return phi_centered.std()

	def getPzNoise(self, psi):
		# FIXME: check that this formula is correct
		# (may need some additional terms like <N^2>)
		n = self._env.fromDevice(psi.density_meter.getNPerEnsemble())
		Pz = (n[0] - n[1]) / (n[0] + n[1])
		return Pz.std()

	def getNstddev(self, psi):
		# FIXME: probably need to add modifier here (M^2 / 2)
		n = self._env.fromDevice(psi.density_meter.getNPerEnsemble())
		return numpy.std(n, axis=1)

	def getEnsembleSums(self, psi):
		"""
		Returns per-ensemble populations and interaction
		"""
		i = self._env.fromDevice(psi.interaction_meter.getIPerEnsemble())
		n = self._env.fromDevice(psi.density_meter.getNPerEnsemble())
		return i, n


def getSpins(i, n1, n2):
	"""Get spin point coordinates on Bloch sphere"""

	# Si for each trajectory
	Si = [i.real, i.imag, 0.5 * (n1 - n2)]
	S = numpy.sqrt(Si[0] ** 2 + Si[1] ** 2 + Si[2] ** 2)
	phi = numpy.arctan2(Si[1], Si[0])
	yps = numpy.arccos(Si[2] / S)

	return phi, yps


def getXiSquared(i, n1, n2):
	"""Get squeezing coefficient; see Yun Li et al, Eur. Phys. J. B 68, 365-381 (2009)"""

	# TODO: some generalization required for >2 components

	Si = [i.real, i.imag, 0.5 * (n1 - n2)] # S values for each trajectory
	avgs = [x.mean() for x in Si] # <S_i>, i=x,y,z

	# \Delta_{ii} = 2 \Delta S_i^2
	deltas = numpy.array([[(x * y + y * x).mean() - 2 * x.mean() * y.mean() for x in Si] for y in Si])

	S = numpy.sqrt(avgs[0] ** 2 + avgs[1] ** 2 + avgs[2] ** 2) # <S>
	phi = numpy.arctan2(avgs[1], avgs[0]) # azimuthal angle of S
	yps = numpy.arccos(avgs[2] / S) # polar angle of S

	sin = numpy.sin
	cos = numpy.cos

	A = (sin(phi) ** 2 - cos(yps) ** 2 * cos(phi) ** 2) * 0.5 * deltas[0, 0] + \
		(cos(phi) ** 2 - cos(yps) ** 2 * sin(phi) ** 2) * 0.5 * deltas[1, 1] - \
		sin(yps) ** 2 * 0.5 * deltas[2, 2] - \
		0.5 * (1 + cos(yps) ** 2) * sin(2 * phi) * deltas[0, 1] + \
		0.5 * sin(2 * yps) * cos(phi) * deltas[2, 0] + \
		0.5 * sin(2 * yps) * sin(phi) * deltas[1, 2]

	B = cos(yps) * sin(2 * phi) * (0.5 * deltas[0, 0] - 0.5 * deltas[1, 1]) - \
		cos(yps) * cos(2 * phi) * deltas[0, 1] - \
		sin(yps) * sin(phi) * deltas[2, 0] + \
		sin(yps) * cos(phi) * deltas[1, 2]

	Sperp_squared = \
		0.5 * (cos(yps) ** 2 * cos(phi) ** 2 + sin(phi) ** 2) * 0.5 * deltas[0, 0] + \
		0.5 * (cos(yps) ** 2 * sin(phi) ** 2 + cos(phi) ** 2) * 0.5 * deltas[1, 1] + \
		0.5 * sin(yps) ** 2 * 0.5 * deltas[2, 2] - \
		0.25 * sin(yps) ** 2 * sin(2 * phi) * deltas[0, 1] - \
		0.25 * sin(2 * yps) * cos(phi) * deltas[2, 0] - \
		0.25 * sin(2 * yps) * sin(phi) * deltas[1, 2] - \
		0.5 * numpy.sqrt(A ** 2 + B ** 2)

	Na = n1.mean()
	Nb = n2.mean()

	return (Na + Nb) * Sperp_squared / (S ** 2)
