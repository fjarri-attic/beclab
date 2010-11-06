from . import helpers

_PRELUDE = """
#define DEFINE_INDEXES unsigned int index = GLOBAL_ID_FLAT, cell_index = index % ${c.cells}
"""

def getEnvWrapper(env_class):

	class Temp(env_class):
		def compileProgram(self, source, constants, **kwds):
			from mako.template import Template
			prelude = Template(_PRELUDE).render(c=constants, **kwds)

			return self.compile(source, double=constants.double,
				prelude=prelude, c=constants, **kwds)

	Temp.__name__ = env_class.__name__
	return Temp


class _Envs:

	def __init__(self):
		try:
			import helpers.cl
			self.cl = getEnvWrapper(helpers.cl.CLEnvironment)
		except:
			pass

		try:
			import helpers.cuda
			self.cuda = getEnvWrapper(helpers.cuda.CUDAEnvironment)
		except:
			pass

		import helpers.cpu
		self.cpu = helpers.cpu.CPUEnvironment

envs = _Envs()

from .model import Model
from .constants import Constants, COMP_1_minus1, COMP_2_1
from .ground_state import GPEGroundState
from .evolution import SplitStepEvolution, SplitStepEvolution2, RungeKuttaEvolution, RK4Evolution
from .pulse import Pulse
from .collectors import AxialProjectionCollector, ParticleNumberCollector, \
	SurfaceProjectionCollector, VisibilityCollector, SliceCollector, \
	PhaseNoiseCollector, PzNoiseCollector, UncertaintyCollector

# FIXME: temporary, just to run simulations on VPAC
try:
	from .datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot, EvolutionPlot
except Exception as e:
	print "Failed to load datahelpers: " + str(e)
