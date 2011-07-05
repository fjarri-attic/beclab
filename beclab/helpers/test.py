import numpy
import unittest
import itertools

from beclab.helpers.cpu import CPUEnvironment
from beclab.helpers.cl import CLEnvironment
from beclab.helpers.cuda import CUDAEnvironment

from beclab.helpers.fft import createFFTPlan
from beclab.helpers.transpose import createTranspose
from beclab.helpers.reduce import createReduce


def getTestArray(shape, dtype, batch):
	large_shape = tuple([shape[0] * batch] + list(shape)[1:])

	if dtype in [numpy.float32, numpy.float64]:
		return numpy.random.randn(*large_shape).astype(dtype)
	else:
		return numpy.random.randn(*large_shape).astype(dtype) + \
			1j * numpy.random.randn(*large_shape).astype(dtype)

def diff(a1, a2):
	return numpy.linalg.norm(a1 - a2) / numpy.linalg.norm(a1)

def createTestcase(tc_class, env):

	class Temp(tc_class):
		def setUp(self):
			self.env = env
			tc_class.setUp(self)

	Temp.__name__ = tc_class.__name__ + "_" + str(env)
	return Temp


class FFTTest(unittest.TestCase):

	testcases = itertools.product((
			(8,), (256,), (8192,),
			(8, 8), (256, 128), (8192, 8),
			(8, 8, 8), (256, 16, 16)
		), (numpy.complex64,), (1, 4, 8))

	def testBuffers(self):
		for shape, dtype, batch in self.testcases:
			data = getTestArray(shape, dtype, batch)

			d_data = self.env.toDevice(data)
			d_transformed = self.env.allocate(data.shape, dtype)
			self.env.copyBuffer(d_data, d_transformed)
			back = self.env.fromDevice(d_transformed)

			self.assert_(diff(data, back) < 1e-6)

	def testForwardInverse(self):

		for shape, dtype, batch in self.testcases:
			data = getTestArray(shape, dtype, batch)
			plan = createFFTPlan(self.env, shape, dtype)

			d_data = self.env.toDevice(data)
			d_transformed = self.env.allocate(data.shape, dtype)
			d_back = self.env.allocate(data.shape, dtype)

			plan.execute(d_data, d_transformed, batch=batch)
			plan.execute(d_transformed, d_back, batch=batch, inverse=True)

			back = self.env.fromDevice(d_back)
			print diff(data, back)
			self.assert_(diff(data, back) < 1e-6)

	def testCompare(self):

		ref_env = CPUEnvironment()
		for shape, dtype, batch in self.testcases:
			data = getTestArray(shape, dtype, batch)
			plan = createFFTPlan(self.env, shape, dtype)
			ref_plan = createFFTPlan(ref_env, shape, dtype)

			d_data = self.env.toDevice(data)
			d_temp = self.env.allocate(data.shape, dtype)

			ref_data = ref_env.toDevice(data)
			ref_temp = ref_env.allocate(data.shape, dtype)

			plan.execute(d_data, d_temp, batch=batch)
			ref_plan.execute(ref_data, ref_temp, batch=batch)

			self.assert_(diff(self.env.fromDevice(d_temp),
				ref_env.fromDevice(ref_temp)) < 1e-6)


class TransposeTest(unittest.TestCase):

	testcases = itertools.product(
		(
			(16, 16), (32, 16), (512, 64),
			(3, 5), (3, 32), (15, 32), (379, 133)
		),
		(numpy.float32, numpy.complex64),
		(1, 3, 15, 16, 17))

	def testDoubleTranspose(self):
		for shape, dtype, batch in self.testcases:

			data = getTestArray(shape, dtype, batch)
			tr = createTranspose(self.env, dtype)
			d_data = self.env.toDevice(data)
			d_temp = self.env.allocate(data.shape, dtype)
			tr(d_data, d_temp, shape[1], shape[0], batch=batch)
			tr(d_temp, d_data, shape[0], shape[1], batch=batch)
			result = self.env.fromDevice(d_data)

			self.assert_(diff(data, result) < 1e-6)

	def testCompare(self):
		ref_env = CPUEnvironment()

		for shape, dtype, batch in self.testcases:

			data = getTestArray(shape, dtype, batch)

			tr = createTranspose(self.env, dtype)
			ref_tr = createTranspose(ref_env, dtype)

			d_data = self.env.toDevice(data)
			d_temp = self.env.allocate(data.shape, dtype)

			ref_data = ref_env.toDevice(data)
			ref_temp = ref_env.allocate(data.shape, dtype)

			tr(d_data, d_temp, shape[1], shape[0], batch=batch)
			ref_tr(ref_data, ref_temp, shape[1], shape[0], batch=batch)

			result = self.env.fromDevice(d_temp)
			ref_result = ref_env.fromDevice(ref_temp)

			self.assert_(diff(result, ref_result) < 1e-6)


class ReduceTest(unittest.TestCase):

	def testFullReduce(self):

		for size in (140, 512 * 512 + 150, 512 * 231):
			for dtype in (numpy.float32, numpy.complex64):
				data = getTestArray((size,), dtype, 1)
				d_data = self.env.toDevice(data)
				reduce = createReduce(self.env, dtype)

				# FIXME: for some reason, error here is very high
				# is it because of the many additions?
				self.assert_(diff(reduce(d_data), numpy.sum(data)) < 1e-3)

	def testPartialReduce(self):
		ref_env = CPUEnvironment()

		tests = (
			(128, 16),
			(128, 1024 * 8),
			(50, 531),
			(40, 24),
			(457, 389)
		)

		for final_length, multiplier in tests:
			for dtype in (numpy.float32, numpy.complex64):
				data = getTestArray((final_length * multiplier,), dtype, 1)
				d_data = self.env.toDevice(data)
				ref_data = ref_env.toDevice(data)

				reduce = createReduce(self.env, dtype)
				ref_reduce = createReduce(ref_env, dtype)

				d_result = reduce(d_data, final_length=final_length)
				ref_result = ref_reduce(ref_data, final_length=final_length)

				self.assert_(diff(self.env.fromDevice(d_result),
					ref_env.fromDevice(ref_result)) < 1e-3)

	def testSparseReduce(self):
		ref_env = CPUEnvironment()

		tests = (
			(128, 16),
			(128, 1024 * 8),
			(50, 531),
			(40, 24),
			(457, 389)
		)

		for final_length, multiplier in tests:
			for dtype in (numpy.float32, numpy.complex64):
				data = getTestArray((final_length * multiplier,), dtype, 1)
				d_data = self.env.toDevice(data)
				ref_data = ref_env.toDevice(data)

				reduce = createReduce(self.env, dtype)
				ref_reduce = createReduce(ref_env, dtype)

				d_result = reduce.sparse(d_data, final_length=final_length)
				ref_result = ref_reduce.sparse(ref_data, final_length=final_length)

				self.assert_(diff(self.env.fromDevice(d_result),
					ref_env.fromDevice(ref_result)) < 1e-3)


if __name__ == '__main__':

	suites = []

	add_suite = lambda tc_class, env: suites.append(
		unittest.TestLoader().loadTestsFromTestCase(createTestcase(tc_class, env)))

	envs = (CUDAEnvironment(), CLEnvironment(), CPUEnvironment())
	tc_classes = (FFTTest, TransposeTest, ReduceTest)

	for env in envs:
		for tc_class in tc_classes:
			add_suite(tc_class, env)

	all = unittest.TestSuite(suites)
	unittest.TextTestRunner(verbosity=1).run(all)

	for env in envs:
		env.release()
