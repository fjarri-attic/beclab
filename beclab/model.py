"""
Default physical world parameters
"""

class Model:
	"""
	Model parameters;
	in SI units, unless explicitly specified otherwise
	"""

	hbar = 1.054571628e-34 # Planck constant
	a0 = 5.2917720859e-11 # Bohr radius

	N = 150000 # number of particles
	m = 1.443160648e-25 # mass of one particle (rubidium-87)

	# scattering lengths, in Bohr radii
	# source:
	# private communication with Servaas Kokkelmans and the paper
	# B. J. Verhaar, E. G. M. van Kempen, and S. J. J.
	# M. F. Kokkelmans, Phys. Rev. A 79, 032711 (2009).
	a11 = 100.4
	a22 = 95.68
	a12 = 98.13

	# Trap frequencies
	fx = 97.6
	fy = 97.6
	fz = 11.96

	# coupling field properties
	detuning = -41 # detuning from hyperfine frequency
	rabi_freq = 350 # Rabi frequency

	hf_freq = 6.834682610904290e9 # 5^2S_{1/2} hyperfine splitting frequency

	# loss terms
	gamma111 = 5.4e-42
	gamma12 = 0.78e-19
	gamma22 = 1.194e-19

	# spatial lattice size
	nvx = 16
	nvy = 16
	nvz = 128

	# number of iterations for integration in mid step
	itmax = 3

	dt_steady = 2e-5 # time step for steady state calculation
	dt_evo = 4e-5 # time step for evolution

	ensembles = 4 # number of ensembles

	border = 1.2 # defines, how big is calculation area as compared to cloud size

	def __init__(self, **kwds):
		for kwd in kwds:
			self.__dict__[kwd] = kwds[kwd]
