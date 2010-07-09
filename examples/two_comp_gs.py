import numpy
import time
import math

from beclab import *

def testTwoCompGS(gpu):

	# preparation
	env = Environment(gpu=gpu)
	constants = Constants(Model(N=150000), double_precision=False if gpu else True)
	gs = GPEGroundState(env, constants)
	sp = SliceCollector(env, constants, pulse=None)

	cloud = gs.createCloud(two_component=True, precision=1e-9)
	sp(0, cloud)

	# render
	times, a_xy, a_yz, b_xy, b_yz = sp.getData()

	a_data = HeightmapData("1 component", a_yz[0].transpose(), xmin=-constants.zmax, xmax=constants.zmax,
		xname="Z, $\\mu$m", yname="Y, $\\mu$m",
		ymin=-constants.ymax, ymax=constants.ymax, zmin=0)
	a_plot = HeightmapPlot(a_data)

	b_data = HeightmapData("2 component", b_yz[0].transpose(), xmin=-constants.zmax, xmax=constants.zmax,
		xname="Z, $\\mu$m", yname="Y, $\\mu$m",
		ymin=-constants.ymax, ymax=constants.ymax, zmin=0)
	b_plot = HeightmapPlot(b_data)

	return a_plot, b_plot

p1, p2 = testTwoCompGS(False)
p1.save("two_comp_gs_cpu_a.pdf")
p2.save("two_comp_gs_cpu_b.pdf")

p1, p2 = testTwoCompGS(True)
p1.save("two_comp_gs_gpu_a.pdf")
p2.save("two_comp_gs_gpu_b.pdf")