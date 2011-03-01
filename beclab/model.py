"""
Default physical world parameters
"""

class Model:
	"""
	Model parameters;
	in SI units, unless explicitly specified otherwise
	"""

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

	# cutoff energy for vacuum noise, in mu_TF(|1,-1>)
	e_cut = 1

	# coupling field properties
	detuning = -41 # detuning from hyperfine frequency
	rabi_freq = 350 # Rabi frequency

	hf_freq = 6.834682610904290e9 # 5^2S_{1/2} hyperfine splitting frequency

	# loss terms (according to M. Egorov, as of 28 Feb 2011; for 44k atoms)
	gamma111 = 5.4e-42
	gamma12 = 1.52e-20
	gamma22 = 7.7e-20

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
