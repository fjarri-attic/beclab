from .fft import createFFTPlan
from .fht import FHT1D, FHT3D, getHarmonicGrid
from .transpose import createTranspose
from .reduce import createReduce
from .misc import PairedCalculation, log2, tile3D
from .typenames import double_precision, single_precision
from .random import createRandom

def createFHTPlan(env, constants, grid, order):

	if grid.dim == 3:
		return FHT3D(env, constants, grid, grid.mshape, order, (grid.lz, grid.ly, grid.lx))
	else:
		return FHT1D(env, constants, grid, grid.mshape[0], order, grid.lz)

