import numpy
import unittest, itertools
import fractions
from numpy.polynomial import Hermite as H

from .misc import tile3D, PairedCalculation
from .transpose import createPermute


def factorial(n):
	res = 1
	for i in xrange(2, n + 1):
		res *= i
	return res

def diff(x, y):
	return numpy.linalg.norm(x - y) / numpy.linalg.norm(x)

def my_hermite(n):
	"""Returns orthonormal Hermite polynomial"""
	def func(x):
		return H([0] * n + [1])(x) / (numpy.pi ** 0.25) / numpy.sqrt(float(2 ** n * factorial(n)))

	return func

def my_h_roots(n):
	"""
	Recursive root finding algorithm, taken from Numerical Recipes.
	More accurate than standard h_roots() from scipy.
	"""

	EPS = 1.0e-16
	PIM4 = numpy.pi ** (-0.25) # 0.7511255444649425
	MAXIT = 20 # Maximum iterations.

	x = numpy.empty(n)
	w = numpy.empty(n)
	m = (n + 1) / 2

	z = 0

	for i in xrange(m):
		if i == 0: # Initial guess for the largest root.
			z = numpy.sqrt(float(2 * n + 1)) - 1.85575 * float(2 * n + 1) ** (-0.16667)
		elif i == 1:
			z -= 1.14 * float(n) ** 0.426 / z
		elif i == 2:
			z = 1.86 * z + 0.86 * x[0]
		elif i == 3:
			z = 1.91 * z + 0.91 * x[1]
		else:
			z = 2.0 * z + x[i - 2]

		for its in xrange(MAXIT):
			p1 = PIM4
			p2 = 0.0
			p3 = 0.0
			for j in xrange(n):
				p3 = p2
				p2 = p1
				p1 = z * numpy.sqrt(2.0 / (j + 1)) * p2 - numpy.sqrt(float(j) / (j + 1)) * p3

			pp = numpy.sqrt(float(2 * n)) * p2
			z1 = z
			z = z1 - p1 / pp
			if abs(z - z1) <= EPS:
				break

		if its >= MAXIT:
			raise Exception("too many iterations in gauher")

		x[n - 1 - i] = z
		x[i] = -z
		w[i] = 2.0 / (pp ** 2)
		w[n - 1 - i] = w[i]

	return x, w

def getHarmonicGrid(N, l, dp=0):
	if (N - 1) * (l + 1) + 1 % 2 == 0:
		points = ((N - 1) * (l + 1) + 1) / 2
	else:
		points = ((N - 1) * (l + 1) + 2) / 2

	# TODO: Population calculated in mode and in x-space is slightly different
	# The more points we take in addition to minimum necessary for precise
	# G.-H. quadrature, the less is the difference.
	# Looks like it is not a bug, just inability to integrate Hermite function
	# in x-space precisely (oscillates too fast maybe?).
	# But this still requres investigation.
	# In addition: with dp != 0 X-M-X transform of TF state
	# gives non-smooth curve. Certainly TF state has some higher harmonics
	# (infinite number of them, to be precise), but why it is smooth when dp = 0?
	points += dp

	roots, weights = my_h_roots(points)

	return roots * numpy.sqrt(2.0 / (l + 1)), \
		weights * numpy.exp(roots ** 2) * numpy.sqrt(2.0 / (l + 1))

def getEigenfunction(n):
	return lambda x: my_hermite(n)(x) * numpy.exp(-(x ** 2) / 2)

def getEigenfunction1D(nx):
	return lambda x: getEigenfunction(nx)(x)

def getEigenfunction3D(nx, ny, nz):
	return lambda x, y, z: getEigenfunction(nx)(x) * \
		getEigenfunction(ny)(y) * getEigenfunction(nz)(z)

def getPMatrix(N, l, dp):
	x, _ = getHarmonicGrid(N, l, dp)

	res = numpy.zeros((N, len(x)))

	for n in range(N):
		phi = getEigenfunction(n)
		res[n, :] = phi(x)

	return res

class FHT1D(PairedCalculation):

	def __init__(self, env, constants, grid, N, order, scale=1, dp=0):
		"""
		N: the maximum number of harmonics
		(i.e. transform returns decomposition on eigenfunctions with numbers 0 .. N)
		order: the order of transformed function (i.e. for f = Psi^2 l = 2)
		(f() cannot have mixed order, i.e. no f() = Psi^2 + Psi)
		"""
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		self.N = N
		self.order = order
		_, w = getHarmonicGrid(N, order, dp)

		self._scalar_dtype = constants.scalar.dtype
		self._complex_dtype = constants.complex.dtype

		self._weights_x = self._env.toDevice(w.astype(self._scalar_dtype))
		self._xshape = (len(w),)

		P = getPMatrix(self.N, self.order, dp).astype(self._scalar_dtype)
		self._P = self._env.toDevice(P)

		# flatten and reshape make memory linear again
		# (transpose() just swaps strides)
		P_tr = P.transpose()
		self._P_tr = self._env.toDevice(P_tr.flatten().reshape(P_tr.shape))

		self._fwd_scale = constants.scalar.cast(numpy.sqrt(scale))
		self._inv_scale = constants.scalar.cast(1.0 / numpy.sqrt(scale))

		self._cached_batch = None
		self._allocateXi(1)

		self._prepare()

	def _allocateXi(self, batch):
		if self._cached_batch != batch:
			self._Xi = self._env.allocate((batch,) + self._xshape, self._complex_dtype)
			self._cached_batch = batch

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernel_template = """
			EXPORTED_FUNC void multiplyTiledCS(int gsize,
				GLOBAL_MEM COMPLEX *res, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *coeffs, int batch)
			{
				LIMITED_BY(gsize);

				SCALAR coeff_val = coeffs[GLOBAL_INDEX % ${size}];
				COMPLEX data_val = data[GLOBAL_INDEX];
				res[GLOBAL_INDEX] = complex_mul_scalar(data_val, coeff_val);
			}

			EXPORTED_FUNC void matrixMulCS(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *m1, GLOBAL_MEM SCALAR *m2, SCALAR scale,
				int w1, int h1, int w2)
			{
				LIMITED_BY(gsize);

				COMPLEX sum = complex_ctr(0, 0);
				int target_x = GLOBAL_INDEX % w2;
				int target_y = GLOBAL_INDEX / w2;

				for(int i = 0; i < w1; i++)
					sum = sum + complex_mul_scalar(m1[target_y * w1 + i], m2[i * w2 + target_x]);

				res[GLOBAL_INDEX] = complex_mul_scalar(sum, scale);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, size=self._xshape[0])

		self._kernel_multiplyTiledCS = self._program.multiplyTiledCS
		self._kernel_matrixMulCS = self._program.matrixMulCS

	def _cpu__kernel_matrixMulCS(self, gsize, res, m1, m2, scale, w1, h1, w2):
		self._env.copyBuffer(numpy.dot(m1, m2), dest=res)
		res *= scale

	def _cpu__kernel_multiplyTiledCS(self, gsize, res, data, coeffs, batch):
		self._env.copyBuffer(data.flat * numpy.tile(coeffs.flat, batch), dest=res)

	def execute(self, data, result, inverse=False, batch=1):
		cast = numpy.int32
		if inverse:
			assert data.shape[-1:] == (self.N,)

			self._kernel_matrixMulCS(batch * self._xshape[0],
				result, data, self._P,
				self._inv_scale,
				cast(self.N), cast(batch), cast(self._xshape[0]))
		else:
			assert data.shape[-1:] == self._xshape
			self._allocateXi(batch)
			self._kernel_multiplyTiledCS(batch * self._xshape[0], self._Xi, data,
				self._weights_x, cast(batch))

			self._kernel_matrixMulCS(batch * self.N,
				result, self._Xi, self._P_tr,
				self._fwd_scale,
				cast(self._xshape[0]), cast(batch), cast(self.N))


class FHT3D(PairedCalculation):

	def __init__(self, env, constants, grid, N, order, scale=(1, 1, 1), dp=(0, 0, 0)):
		"""
		N: the maximum number of harmonics (tuple (Nz, Ny, Nx))
		(i.e. transform returns decomposition on eigenfunctions with numbers 0 .. N - 1)
		order: the order of transformed function (i.e. for f = Psi^2 l = 2)
		(f() cannot have mixed order, i.e. no f() = Psi^2 + Psi)
		"""
		PairedCalculation.__init__(self, env)
		self._constants = constants.copy()
		self._grid = grid.copy()

		complex_cast = lambda x: x.astype(self._constants.complex.dtype)
		scalar_cast = lambda x: x.astype(self._constants.scalar.dtype)
		self._complex_dtype = self._constants.complex.dtype
		self._scalar_cast = constants.scalar.cast

		self.N = N
		self.order = order
		scale = numpy.sqrt(scale[0] * scale[1] * scale[2])
		self._fwd_scale = constants.scalar.cast(scale)
		self._inv_scale = constants.scalar.cast(1.0 / scale)

		_, wx = getHarmonicGrid(N[2], order, dp[2])
		_, wy = getHarmonicGrid(N[1], order, dp[1])
		_, wz = getHarmonicGrid(N[0], order, dp[0])

		self._xshape = (len(wz), len(wy), len(wx))
		wx, wy, wz = tile3D(wx, wy, wz)

		self._weights = self._env.toDevice(scalar_cast(wx * wy * wz))

		Px = scalar_cast(getPMatrix(self.N[2], self.order, dp[2]))
		Py = scalar_cast(getPMatrix(self.N[1], self.order, dp[1]))
		Pz = scalar_cast(getPMatrix(self.N[0], self.order, dp[0]))

		Px_tr = Px.transpose()
		Py_tr = Py.transpose()
		Pz_tr = Pz.transpose()

		self._Px = self._env.toDevice(Px)
		self._Py = self._env.toDevice(Py)
		self._Pz = self._env.toDevice(Pz)
		self._Px_tr = self._env.toDevice(Px_tr.flatten().reshape(Px_tr.shape))
		self._Py_tr = self._env.toDevice(Py_tr.flatten().reshape(Py_tr.shape))
		self._Pz_tr = self._env.toDevice(Pz_tr.flatten().reshape(Pz_tr.shape))

		self._cached_batch = None
		self._allocateTempArrays(1)
		self._permute = createPermute(env, constants.complex.dtype)

		self._prepare()

	def _allocateTempArrays(self, batch):
		if self._cached_batch != batch:
			self._temp1 = self._env.allocate((batch,) + self._xshape, self._complex_dtype)
			self._temp2 = self._env.allocate((batch,) + self._xshape, self._complex_dtype)
			self._cached_batch = batch

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernel_template = """
			EXPORTED_FUNC void multiplyTiledCS(int gsize,
				GLOBAL_MEM COMPLEX *res, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR *coeffs, int batch)
			{
				LIMITED_BY(gsize);
				SCALAR coeff_val = coeffs[GLOBAL_INDEX % ${size}];
				COMPLEX data_val = data[GLOBAL_INDEX];
				res[GLOBAL_INDEX] = complex_mul_scalar(data_val, coeff_val);
			}

			EXPORTED_FUNC void multiplyConstantCS(int gsize,
				GLOBAL_MEM COMPLEX *res, GLOBAL_MEM COMPLEX *data,
				GLOBAL_MEM SCALAR coeff, int batch)
			{
				LIMITED_BY(gsize);
				COMPLEX data_val = data[GLOBAL_INDEX];
				res[GLOBAL_INDEX] = complex_mul_scalar(data_val, coeff);
			}

			EXPORTED_FUNC void matrixMulCS(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *m1, GLOBAL_MEM SCALAR *m2,
				int w1, int h1, int w2)
			{
				LIMITED_BY(gsize);
				int output_index = GLOBAL_INDEX;

				COMPLEX sum = complex_ctr(0, 0);
				int target_x = output_index % w2;
				int target_y = output_index / w2;

				for(int i = 0; i < w1; i++)
					sum = sum + complex_mul_scalar(m1[target_y * w1 + i], m2[i * w2 + target_x]);

				res[output_index] = sum;
			}

			EXPORTED_FUNC void matrixMulCS2(int gsize, GLOBAL_MEM COMPLEX *res,
				GLOBAL_MEM COMPLEX *m1, GLOBAL_MEM SCALAR *m2, SCALAR coeff,
				int w1, int h1, int w2)
			{
				LIMITED_BY(gsize);
				int output_index = GLOBAL_INDEX;

				COMPLEX sum = complex_ctr(0, 0);
				int target_x = output_index % w2;
				int target_y = output_index / w2;

				for(int i = 0; i < w1; i++)
					sum = sum + complex_mul_scalar(m1[target_y * w1 + i], m2[i * w2 + target_x]);

				res[output_index] = complex_mul_scalar(sum, coeff);
			}
		"""

		self._program = self._env.compileProgram(kernel_template, self._constants,
			self._grid, size=self._xshape[0] * self._xshape[1] * self._xshape[2])

		self._kernel_multiplyTiledCS = self._program.multiplyTiledCS
		self._kernel_multiplyConstantCS = self._program.multiplyConstantCS
		self._kernel_matrixMulCS = self._program.matrixMulCS
		self._kernel_matrixMulCS2 = self._program.matrixMulCS2

	def _cpu__kernel_matrixMulCS(self, gsize, res, m1, m2, w1, h1, w2):
		m1 = m1.flat[:w1*h1].reshape(h1, w1)
		m2 = m2.flat[:w1*w2].reshape(w1, w2)
		res.flat[:gsize] = numpy.dot(m1, m2).flat

	def _cpu__kernel_matrixMulCS2(self, gsize, res, m1, m2, coeff, w1, h1, w2):
		m1 = m1.flat[:w1*h1].reshape(h1, w1)
		m2 = m2.flat[:w1*w2].reshape(w1, w2)
		res.flat[:gsize] = numpy.dot(m1, m2).flat * coeff

	def _cpu__kernel_multiplyTiledCS(self, gsize, res, data, coeffs, batch):
		self._env.copyBuffer(data.flat * numpy.tile(coeffs.flat, batch), dest=res)

	def _cpu__kernel_multiplyConstantCS(self, gsize, res, data, coeff, batch):
		self._env.copyBuffer(data * coeff, dest=res)

	def execute(self, data, result, inverse=False, batch=1):
		Mz, My, Mx = self._xshape
		Nz, Ny, Nx = self.N

		cast = numpy.int32

		self._allocateTempArrays(batch)

		if inverse:
			assert data.shape[-3:] == self.N
			self._permute(data, self._temp1, (Nz, Ny, Nx), batch=batch)
			self._kernel_matrixMulCS(batch * Ny * Nx * Mz,
				self._temp2, self._temp1, self._Pz,
				cast(Nz), cast(batch * Ny * Nx), cast(Mz))
			self._permute(self._temp2, self._temp1, (Ny, Nx, Mz), batch=batch)
			self._kernel_matrixMulCS(batch * Nx * Mz * My,
				self._temp2, self._temp1, self._Py,
				cast(Ny), cast(batch * Nx * Mz), cast(My))
			self._permute(self._temp2, self._temp1, (Nx, Mz, My), batch=batch)
			self._kernel_matrixMulCS(batch * Mz * My * Mx,
				result, self._temp1, self._Px,
				cast(Nx), cast(batch * Mz * My), cast(Mx))
			self._kernel_multiplyConstantCS(batch * Mz * My * Mx, result, result,
				self._scalar_cast(self._inv_scale), cast(batch))

			"""
			res = numpy.dot(
				numpy.transpose(data, axes=(0, 2, 3, 1)), # batch, Ny, Nx, Nz
				self._Pz, # Nz, Mz
			) # batch, Ny, Nx, Mz
			res = numpy.dot(
				numpy.transpose(res, axes=(0, 2, 3, 1)), # batch, Nx, Mz, Ny
				self._Py, # Ny, My
			) # batch, Nx, Mz, My
			res = numpy.dot(
				numpy.transpose(res, axes=(0, 2, 3, 1)), # batch, Mz, My, Nx
				self._Px, # Nx, Mx
			) # batch, Mz, My, Mx

			result.flat[:] = (res * self._inv_scale).flat
			"""
		else:
			assert data.shape[-3:] == (Mz, My, Mx)
			self._kernel_multiplyTiledCS(batch * Mz * My * Mx, self._temp2,
				data, self._weights, cast(batch))
			self._permute(self._temp2, self._temp1, (Mz, My, Mx), batch=batch)
			self._kernel_matrixMulCS(batch * My * Mx * Nz,
				self._temp2, self._temp1, self._Pz_tr,
				cast(Mz), cast(batch * My * Mx), cast(Nz))
			self._permute(self._temp2, self._temp1, (My, Mx, Nz), batch=batch)
			self._kernel_matrixMulCS(batch * Mx * Nz * Ny,
				self._temp2, self._temp1, self._Py_tr,
				cast(My), cast(batch * Mx * Nz), cast(Ny))
			self._permute(self._temp2, self._temp1, (Mx, Nz, Ny), batch=batch)
			self._kernel_matrixMulCS2(batch * Nz * Ny * Nx,
				result, self._temp1, self._Px_tr, self._scalar_cast(self._fwd_scale),
				cast(Mx), cast(batch * Nz * Ny), cast(Nx))
			# FIXME: for some reason this function damages nearby buffers;
			# multiplication in matrixMul seems to work well.
			#self._kernel_multiplyConstantCS(Nz * Ny * Nx, result, result,
			#	self._scalar_cast(self._fwd_scale), cast(batch))

			"""
			Xi = self.getXiVector(data, batch=batch) # batch, Mz, My, Mx
			res = numpy.dot(
				numpy.transpose(Xi, axes=(0, 2, 3, 1)), # batch, My, Mx, Mz
				self._Pz_tr, # Mz, Nz
			) # batch, My, Mx, Nz
			res = numpy.dot(
				numpy.transpose(res, axes=(0, 2, 3, 1)), # batch, Mx, Nz, My
				self._Py_tr, # My, Ny
			) # batch, Mx, Nz, Ny
			res = numpy.dot(
				numpy.transpose(res, axes=(0, 2, 3, 1)), # batch, Nz, Ny, Mx
				self._Px_tr, # Mx, Nx
			) # batch, Nz, Ny, Nx

			result.flat[:] = (res * self._fwd_scale).flat
			"""


class TestFunction:

	def __init__(self, N, order, coefficients=None):
		self.order = order
		if isinstance(N, int):
			self.N = (N,)
		else:
			self.N = N

		if coefficients is None:
			self.coefficients = numpy.random.rand(*(self.N))
		else:
			self.coefficients = numpy.array(coefficients).astype(numpy.float64)

		if len(self.N) == 1:
			self._efs = [getEigenfunction1D(n) for n in xrange(self.N[0])]
		elif len(self.N) == 3:
			self._efs = [getEigenfunction3D(nx, ny, nz)
				for nz, ny, nx in itertools.product(
					xrange(self.N[0]), xrange(self.N[1]), xrange(self.N[2])
				)
			]
		else:
			raise Exception("Unsupported number of dimensions")

	def __call__(self, *x):
		if len(x) == 3:
			x = tile3D(*x)

		res = numpy.zeros_like(x[0])
		for ef, c in zip(self._efs, self.coefficients.flat):
			res += ef(*x) * c

		return res ** self.order

	def test(self, results, *x):
		reference = self(*x)
		return diff(reference, results.reshape(reference.shape))

	def test_coeffs(self, C):
		return diff(C.reshape(self.coefficients.shape), self.coefficients)


class Test1D(unittest.TestCase):

	def testOrder1(self):

		eps = 1e-11

		print "\nChecking that harmonic decomposition gives right coefficients:"

		for N in (1, 3, 5, 40, 60):
			for i in xrange(1):
				f = TestFunction(N, 1)
				plan = FHT1D(N, 1)

				C = plan.execute(f(plan.grid_x))
				d_fw = f.test_coeffs(C)

				if not (N == 40 or N == 60):
					self.assert_(d_fw < eps)

				f_back = plan.execute(C, inverse=True)
				d_back = f.test(f_back, plan.grid_x)

				if N == 40 or N == 60:
					print "N = {N}, diff forward = {d_fw}, diff back = {d_back}".format(
						N=N, d_fw=d_fw, d_back=d_back)
				else:
					self.assert_(d_back < eps)

	def testHighOrderForward(self):

		eps = 1e-11

		print "\nChecking that with increased number of modes previous terms do not change"
		for order in (2, 3):
			print "Order: " + str(order)
			for N in (20, 40, 60):
				for i in xrange(1):
					f = TestFunction(N, order)
					p1 = FHT1D(N, order)
					p2 = FHT1D(N + 1, order)

					C1 = p1.execute(f(p1.grid_x))
					C2 = p2.execute(f(p2.grid_x))

					d = diff(C1.reshape(N), C2.reshape(N + 1)[:-1])
					print "N = {N}, diff = {d}".format(N=N, d=d)

					# check that decomposition with N + 1 modes has the same 0 .. N modes
					#self.assert_(diff(C1.reshape(N), C2.reshape(N + 1)[:-1]) < eps)

	def testHighOrderBack(self):

		# Psi^2 or ^3 has infinite numbers of harmonics, but we still can count on
		# certain precision of restored function.
		eps = 0.1

		print "\nChecking that F^-1[F[Psi^l]] ~ Psi^l:"
		for order in (2, 3):
			for N in (5, 6, 7, 40, 60):
				f = TestFunction(N, order, coefficients=[1] + [0] * N)
				p = FHT1D(N, order)

				C = p.execute(f(p.grid_x))
				f_back = p.execute(C, inverse=True)
				d = f.test(f_back, p.grid_x)

				if N == 40 or N == 60:
					print "N = {N}, order = {l}, diff = {d}".format(N=N, l=order, d=d)
				else:
					self.assert_(d < eps)

	def testBatch(self):
		eps = 1e-14
		N = 5

		coeffs = numpy.arange(N * 2).reshape(2, N)

		fa = TestFunction(N, 1, coefficients=coeffs[0, :])
		fb = TestFunction(N, 1, coefficients=coeffs[1, :])
		p = FHT1D(N, 1)
		f2 = numpy.concatenate([fa(p.grid_x), fb(p.grid_x)])

		C = p.execute(f2, batch=2)
		self.assert_(C.shape == coeffs.shape)
		self.assert_(diff(C, coeffs) < eps)

		f_back = p.execute(C, batch=2, inverse=True)
		self.assert_(f_back.shape == (2, len(p.grid_x)))
		self.assert_(diff(f_back, f2.reshape(f_back.shape)) < eps)

	def testRoots(self):

		print "\nChecking that roots make polynomial zero:"
		for N in (5, 20, 40):
			x, _ = my_h_roots(N)
			f = my_hermite(N)
			vals = f(x)
			print "N = {N}, min = {vmin}, max = {vmax}".format(N=N,
				vmin=numpy.abs(vals).min(), vmax=numpy.abs(vals).max())


class Test3D(unittest.TestCase):

	def testOrder1(self):

		eps = 1e-11

		for N in ((1, 1, 1), (2, 3, 4), (5, 6, 7)):
			for i in xrange(3):
				f = TestFunction(N, 1)
				plan = FHT3D(N, 1)

				C = plan.execute(f(plan.grid_x, plan.grid_y, plan.grid_z))
				self.assert_(f.test_coeffs(C) < eps)

				f_back = plan.execute(C, inverse=True)
				self.assert_(f.test(f_back, plan.grid_x, plan.grid_y, plan.grid_z) < eps)

	def testHighOrderForward(self):

		eps = 1e-10

		for order in (2, 3):
			for N in ((1, 1, 1), (2, 3, 4), (5, 6, 7)):
				for i in xrange(3):
					f = TestFunction(N, order)
					big_N = tuple(x + 1 for x in N)

					p1 = FHT3D(N, order)
					p2 = FHT3D(big_N, order)

					C1 = p1.execute(f(p1.grid_x, p1.grid_y, p1.grid_z))
					C2 = p2.execute(f(p2.grid_x, p2.grid_y, p2.grid_z))

					# check that decomposition with N + 1 modes has the same 0 .. N modes
					self.assert_(diff(C1, C2[0,:-1,:-1,:-1]) < eps)

	def testHighOrderBack(self):

		# Psi^2 or ^3 has infinite numbers of harmonics, but we still can count on
		# certain precision of restored function.
		eps = 0.1

		for order in (2, 3):
			for N in ((10, 10, 10),):
				f = TestFunction(N, order, coefficients=[1] + [0] * (N[0] * N[1] * N[2] - 1))
				p = FHT3D(N, order)

				C = p.execute(f(p.grid_x, p.grid_y, p.grid_z))
				f_back = p.execute(C, inverse=True)

				self.assert_(f.test(f_back, p.grid_x, p.grid_y, p.grid_z) < eps)

	def testBatch(self):
		eps = 1e-14
		N = (2, 3, 4)

		coeffs = numpy.arange(N[0] * N[1] * N[2] * 2).reshape(2, N[0] * N[1] * N[2])

		fa = TestFunction(N, 1, coefficients=coeffs[0, :])
		fb = TestFunction(N, 1, coefficients=coeffs[1, :])
		p = FHT3D(N, 1)
		f2 = numpy.concatenate([fa(p.grid_x, p.grid_y, p.grid_z), fb(p.grid_x, p.grid_y, p.grid_z)])

		C = p.execute(f2, batch=2)
		self.assert_(C.shape == tuple([2] + list(N)))
		self.assert_(diff(C, coeffs.reshape(C.shape)) < eps)

		f_back = p.execute(C, batch=2, inverse=True)
		self.assert_(f_back.shape == (2, len(p.grid_z), len(p.grid_y), len(p.grid_x)))
		self.assert_(diff(f_back, f2.reshape(f_back.shape)) < eps)


if __name__ == '__main__':
	test1D = unittest.TestLoader().loadTestsFromTestCase(Test1D)
	test3D = unittest.TestLoader().loadTestsFromTestCase(Test3D)
	unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite([test1D, test3D]))
