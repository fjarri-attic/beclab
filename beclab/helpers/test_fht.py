import numpy
import unittest
import itertools

from beclab.helpers.cpu import CPUEnvironment
from beclab.helpers.cl import CLEnvironment
from beclab.helpers.cuda import CUDAEnvironment

from beclab.helpers import createFHTPlan
from beclab.helpers.fht import getHarmonicGrid, getEigenfunction1D, getEigenfunction3D, \
	my_h_roots, my_hermite, diff


def createTestcase(tc_class, env):

	class Temp(tc_class):
		def setUp(self):
			self.env = env
			tc_class.setUp(self)

	Temp.__name__ = tc_class.__name__ + "_" + str(env)
	return Temp


class TestFunction:

	def __init__(self, env, N, order, coefficients=None):
		self.env = env
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

		if len(self.N) == 1:
			x = getHarmonicGrid(self.N[0], order)

			self.data = numpy.zeros_like(x)
			for ef, c in zip(self._efs, self.coefficients.flat):
				self.data += ef(x) * c
		else:
			x = getHarmonicGrid(self.N[2], order)
			y = getHarmonicGrid(self.N[1], order)
			z = getHarmonicGrid(self.N[0], order)
			x, y, z = tile3D(*x)

			self.data = numpy.zeros_like(x)
			for ef, c in zip(self._efs, self.coefficients.flat):
				self.data += ef(x, y, z) * c

		self.data = self.env.toDevice(self.data ** self.order)

	def test(self, results):
		return diff(self.env.fromDevice(self.data),
			self.env.fromDevice(results).reshape(self.data.shape))

	def test_coeffs(self, C):
		return diff(self.env.fromDevice(C).reshape(self.coefficients.shape), self.coefficients)


class Test1D(unittest.TestCase):

	def testOrder1(self):

		eps = 1e-11

		print "\nChecking that harmonic decomposition gives right coefficients:"

		for N in (1, 3, 5, 40, 60):
			f = TestFunction(self.env, N, 1)
			plan = createFHTPlan(self.env, (N,), 1)

			C = plan.execute(f.data)
			d_fw = f.test_coeffs(C)

			if not (N == 40 or N == 60):
				self.assert_(d_fw < eps)

			f_back = plan.execute(C, inverse=True)
			d_back = f.test(f_back)

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
				f1 = TestFunction(self.env, N, order)
				f2 = TestFunction(self.env, N + 1, order)
				p1 = createFHTPlan(self.env, N, order)
				p2 = createFHTPlan(self.env, N + 1, order)

				C1 = p1.execute(f1.data)
				C2 = p2.execute(f2.data)

				d = diff(self.env.fromDevice(C1).reshape(N),
					self.env.fromDevice(C2).reshape(N + 1)[:-1])
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
				f = TestFunction(self.env, N, order, coefficients=[1] + [0] * N)
				p = createFHTPlan(self.env, N, order)

				C = p.execute(f.data)
				f_back = p.execute(C, inverse=True)
				d = f.test(f_back)

				if N == 40 or N == 60:
					print "N = {N}, order = {l}, diff = {d}".format(N=N, l=order, d=d)
				else:
					self.assert_(d < eps)

	def testBatch(self):
		eps = 1e-14
		N = 5

		coeffs = numpy.arange(N * 2).reshape(2, N)

		fa = TestFunction(self.env, N, 1, coefficients=coeffs[0, :])
		fb = TestFunction(self.env, N, 1, coefficients=coeffs[1, :])
		p = createFHTPlan(self.env, N, 1)
		f2 = self.env.toDevice(numpy.concatenate(
			[self.env.fromDevice(fa.data), self.env.fromDevice(fb.data)]))

		C = p.execute(f2, batch=2)
		self.assert_(C.shape == coeffs.shape)
		self.assert_(diff(self.env.fromDevice(C), coeffs) < eps)

		f_back = p.execute(C, batch=2, inverse=True)
		self.assert_(f_back.shape == (2, len(p.grid_x)))
		self.assert_(diff(self.env.fromDevice(f_back),
			self.env.fromDevice(f2).reshape(f_back.shape)) < eps)

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
	suites = []

	add_suite = lambda tc_class, env: suites.append(
		unittest.TestLoader().loadTestsFromTestCase(createTestcase(tc_class, env)))

	envs = (CUDAEnvironment(), CLEnvironment(), CPUEnvironment())
	tc_classes = (Test1D,)

	for env in envs:
		for tc_class in tc_classes:
			add_suite(tc_class, env)

	all = unittest.TestSuite(suites)
	unittest.TextTestRunner(verbosity=1).run(all)

	for env in envs:
		env.release()