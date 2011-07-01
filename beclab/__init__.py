from . import helpers

_PRELUDE = """
#define DEFINE_INDEXES unsigned int index = GLOBAL_ID_FLAT, cell_index = index % ${g.size}, ensemble = index / ${g.size}
"""

def getEnvWrapper(env_class):

	class Temp(env_class):
		def compileProgram(self, source, constants, grid, **kwds):
			from mako.template import Template
			prelude = Template(_PRELUDE).render(c=constants, g=grid, **kwds)

			return self.compile(source, double=constants.double,
				prelude=prelude, c=constants, g=grid, **kwds)

	Temp.__name__ = env_class.__name__
	return Temp


class _Envs:

	def __init__(self):
		try:
			import helpers.cuda
			self.cuda = getEnvWrapper(helpers.cuda.CUDAEnvironment)
		except:
			pass

#		try:
#			import helpers.cl
#			self.cl = getEnvWrapper(helpers.cl.CLEnvironment)
#		except:
#			pass

		import helpers.cpu
		self.cpu = helpers.cpu.CPUEnvironment

envs = _Envs()

from .model import Model
from .constants import Constants, UniformGrid, HarmonicGrid
from .ground_state import GPEGroundState, TFGroundState
#from .evolution import SplitStepEvolution
#from .pulse import Pulse
#from .collectors import AxialProjectionCollector, ParticleNumberCollector, \
#	SurfaceProjectionCollector, VisibilityCollector, SliceCollector, \
#	PhaseNoiseCollector, PzNoiseCollector, UncertaintyCollector, SpinCloudCollector, \
#	AnalyticNoiseCollector

# FIXME: temporary, just to run simulations on VPAC
try:
	from .datahelpers import Data, XYData, HeightmapData, XYPlot, HeightmapPlot, EvolutionPlot
except Exception as e:
	print "Failed to load datahelpers: " + str(e)
