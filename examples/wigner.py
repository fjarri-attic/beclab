import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentEvolution, Pulse
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

constants = Constants(Model(N=30000, detuning=-41, nvx=8, nvy=8, nvz=64, ensembles=16), double_precision=True)
env = Environment(gpu=False)
evolution = TwoComponentEvolution(env, constants)
a = VisibilityCollector(env, constants, verbose=True)
b = ParticleNumberCollector(env, constants, verbose=True)
c = PhaseNoiseCollector(env, constants)

gs = GPEGroundState(env, constants)
pulse = Pulse(env, constants)

cloud = gs.createCloud()

cloud.toWigner()

evolution.run(cloud, 0.02, callbacks=[], callback_dt=1)
pulse.halfPi(cloud)

t1 = time.time()
evolution.run(cloud, 0.4, callbacks=[c], callback_dt=0.002)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

#times, Na, Nb, N = b.getData()
#XYPlot([XYData("test", times, (Na-Nb)/N, ymin=-1, ymax=1, xname="Time, s")]).save('test.pdf')

times, vis = c.getData()
vis = XYData("noise", times, vis, ymin=-10, ymax=10,
	xname="Time, ms", yname="Visibility")
vis = XYPlot([vis])
vis.save('test.pdf')
