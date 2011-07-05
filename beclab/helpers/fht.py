import numpy
import unittest, itertools
import fractions
from numpy.polynomial import Hermite as H

from .misc import tile3D, PairedCalculation


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

def getHarmonicGrid(N, l):
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
	#points += 10

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

def getPMatrix(N, l):
	x, _ = getHarmonicGrid(N, l)

	res = numpy.zeros((N, len(x)))

	for n in range(N):
		phi = getEigenfunction(n)
		res[n, :] = phi(x)

	return res

class FHT1D(PairedCalculation):

	def __init__(self, env, constants, grid, N, order, scale=1):
		"""
		N: the maximum number of harmonics
		(i.e. transform returns decomposition on eigenfunctions with numbers 0 .. N)
		order: the order of transformed function (i.e. for f = Psi^2 l = 2)
		(f() cannot have mixed order, i.e. no f() = Psi^2 + Psi)
		"""
		PairedCalculation.__init__(self, env)
		self._constants = constants
		self._grid = grid

		self.N = N
		self.order = order
		_, w = getHarmonicGrid(N, order)

		self._scalar_dtype = constants.scalar.dtype
		self._complex_dtype = constants.complex.dtype

		self._weights_x = self._env.toDevice(w.astype(self._scalar_dtype))
		self._xshape = (len(w),)

		P = getPMatrix(self.N, self.order).astype(self._scalar_dtype)
		self._P = self._env.toDevice(P)

		# flatten and reshape make memory linear again
		# (transpose() just swaps strides)
		self._P_tr = self._env.toDevice(P.transpose().flatten().reshape(50, 50))

		self._fwd_scale = constants.scalar.cast(numpy.sqrt(scale))
		self._inv_scale = constants.scalar.cast(1.0 / numpy.sqrt(scale))

		self._allocateXi(1)

		self._prepare()

	def _allocateXi(self, batch):
		if not hasattr(self, '_Xi') or self._Xi.shape[0] != batch:
			self._Xi = self._env.allocate((batch,) + self._xshape, self._complex_dtype)

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		pass

	def _cpu__kernel_dot(self, _, res, m1, m2, scale):
		self._env.copyBuffer(numpy.dot(m1, m2), dest=res)
		res *= scale

	def _cpu__kernel_multiply(self, _, res, data, coeffs, batch):
		self._env.copyBuffer(data.flat * numpy.tile(coeffs.flat, batch), dest=res)

	def execute(self, data, result, inverse=False, batch=1):
		if inverse:
			assert data.shape == (batch, self.N)
			self._kernel_dot(None, result, data, self._P, self._inv_scale)
		else:
			assert data.shape == (batch,) + self._xshape
			self._allocateXi(batch)
			self._kernel_multiply(None, self._Xi, data, self._weights_x, batch)
			self._kernel_dot(None, result, self._Xi, self._P_tr, self._fwd_scale)


class FHT3D:

	def __init__(self, N, order, scale=(1, 1, 1)):
		"""
		N: the maximum number of harmonics (tuple (Nz, Ny, Nx))
		(i.e. transform returns decomposition on eigenfunctions with numbers 0 .. N - 1)
		order: the order of transformed function (i.e. for f = Psi^2 l = 2)
		(f() cannot have mixed order, i.e. no f() = Psi^2 + Psi)
		"""

		self.N = N
		self.order = order
		self.scale_coeff = numpy.sqrt(scale[0] * scale[1] * scale[2])
		self.grid_x, self._weights_x = getHarmonicGrid(N[2], order)
		self.grid_y, self._weights_y = getHarmonicGrid(N[1], order)
		self.grid_z, self._weights_z = getHarmonicGrid(N[0], order)

		self.Px = getPMatrix(self.N[2], self.order)
		self.Py = getPMatrix(self.N[1], self.order)
		self.Pz = getPMatrix(self.N[0], self.order)

	def getXiVector(self, data, batch=1):

		Mx = len(self.grid_x)
		My = len(self.grid_y)
		Mz = len(self.grid_z)

		# prepare 1D weight arrays tiled in 3D
		wx = numpy.tile(self._weights_x, My * Mz).reshape(Mz, My, Mx)
		wy = numpy.transpose(numpy.tile(self._weights_y, Mx * Mz).reshape(Mz, Mx, My), axes=(0, 2, 1))
		wz = numpy.transpose(numpy.tile(self._weights_z, My * Mx).reshape(Mx, My, Mz), axes=(2, 1, 0))

		tile = lambda x: numpy.tile(x, (batch, 1, 1)).reshape(batch, Mz, My, Mx)

		return tile(wx) * tile(wy) * tile(wz) * data

	def execute(self, data, result, inverse=False, batch=1):
		Mz, My, Mx = len(self.grid_z), len(self.grid_y), len(self.grid_x)
		Nz, Ny, Nx = self.N

		if inverse:
			data = data.reshape(batch, Nz, Ny, Nx)

			res = numpy.dot(
				self.Pz.transpose(),
				numpy.transpose(data, axes=(1, 2, 3, 0)).reshape(Nz, batch * Nx * Ny)
			).reshape(Mz, Ny, Nx, batch)
			res = numpy.dot(
					self.Py.transpose(),
					numpy.transpose(res, axes=(1, 0, 2, 3)).reshape(Ny, Mz * Nx * batch)
				).reshape(My, Mz, Nx, batch)
			res = numpy.transpose(
				numpy.dot(
					self.Px.transpose(),
					numpy.transpose(res, axes=(2, 0, 1, 3)).reshape(Nx, My * Mz * batch)
				).reshape(Mx, My, Mz, batch),
				axes=(3, 2, 1, 0)
			)

			result.flat[:] = (res / self.scale_coeff).flat

		else:
			data = data.reshape(batch, Mz, My, Mx)

			Xi = self.getXiVector(data, batch=batch)

			res = numpy.dot(
				self.Pz,
				numpy.transpose(Xi, axes=(1, 2, 3, 0)).reshape(Mz, batch * Mx * My)
			).reshape(Nz, My, Mx, batch)
			res = numpy.dot(
					self.Py, numpy.transpose(res, axes=(1, 0, 2, 3)).reshape(My, Nz * Mx * batch)
				).reshape(Ny, Nz, Mx, batch)
			res = numpy.transpose(
				numpy.dot(
					self.Px, numpy.transpose(res, axes=(2, 0, 1, 3)).reshape(Mx, Ny * Nz * batch)
				).reshape(Nx, Ny, Nz, batch),
				axes=(3, 2, 1, 0)
			)

			result.flat[:] = (res * self.scale_coeff).flat


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
