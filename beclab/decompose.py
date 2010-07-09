import numpy
import math


def getTakagiDecomposition(psi1, psi2, k12, k22):
	alpha = psi1 / psi2
	beta = 2 * k22 / k12
	coeff = (abs(psi2) ** 2) * k12

	A = numpy.array([[0, 1, 0, alpha], [1, 0, numpy.conj(alpha), 0],
		[0, numpy.conj(alpha), 0, abs(alpha) ** 2 + beta],
		[alpha, 0, abs(alpha) ** 2 + beta, 0]])

	x = abs(alpha) ** 2 + beta + 1
	lambdas = [
		0.5 * (x ** 2 - 2 * beta - x * math.sqrt(x ** 2 - 4 * beta)),
		0.5 * (x ** 2 - 2 * beta - x * math.sqrt(x ** 2 - 4 * beta)),
		0.5 * (x ** 2 - 2 * beta + x * math.sqrt(x ** 2 - 4 * beta)),
		0.5 * (x ** 2 - 2 * beta + x * math.sqrt(x ** 2 - 4 * beta))
	]

	vectors = [
		numpy.array([0, (-x + 2 - math.sqrt(x ** 2 - 4 * beta)) / (2.0 * alpha), 0, 1]),
		numpy.array([(-x + 2 - math.sqrt(x ** 2 - 4 * beta)) / (2.0 * numpy.conj(alpha)), 0, 1, 0]),
		numpy.array([0, (-x + 2 + math.sqrt(x ** 2 - 4 * beta)) / (2.0 * alpha), 0, 1]),
		numpy.array([(-x + 2 + math.sqrt(x ** 2 - 4 * beta)) / (2.0 * numpy.conj(alpha)), 0, 1, 0])]

	def normalize(v):
		return v / math.sqrt(numpy.dot(v, v.conj()))

	v_transp = numpy.array([
		normalize(numpy.dot(A, vectors[0].conj()) + numpy.sqrt(lambdas[0]) * vectors[0]),
		normalize(numpy.dot(A, vectors[1].conj()) - numpy.sqrt(lambdas[0]) * vectors[1]),
		normalize(numpy.dot(A, vectors[2].conj()) + numpy.sqrt(lambdas[2]) * vectors[2]),
		normalize(numpy.dot(A, vectors[3].conj()) - numpy.sqrt(lambdas[2]) * vectors[3])
	])

	v = v_transp.transpose()
	l = numpy.sqrt(lambdas)
	l[1] = -l[1]
	l[3] = -l[3]
	D = numpy.diag(numpy.sqrt(l.astype(numpy.complex64)))

	B = numpy.dot(v, D) * math.sqrt(coeff)

	return B
