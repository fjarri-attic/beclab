import numpy
import time
import itertools

from beclab import *
from beclab.constants import buildProjectorMask
from beclab.helpers.misc import log2


def testCutoff(*args):
	env = envs.cuda(device_num=1)
	try:
		return runTest(env, *args)
	finally:
		env.release()

def runTest(env, dim, grid_type, prop_type, use_cutoff, use_big_grid):

	# additional parameters
	constants_kwds = {
		'1d': dict(use_effective_area=True, fx=42e3, fy=42e3, fz=90),
		'3d': {}
	}[dim]

	# total number of atoms in ground state
	total_N = {
		'1d': 60,
		'3d': 50000
	}[dim]

	# number of lattice points (for no cutoff mode)
	shape = {
		('1d', 'uniform'): (64,),
		('3d', 'uniform'): (128, 8, 8),
		('1d', 'harmonic'): (50,),
		('3d', 'harmonic'): (50, 10, 10)
	}[(dim, grid_type)]

	# time step for split-step propagation
	ss_dt = {
		'1d': 1e-6,
		'3d': 1e-5
	}[dim]

	e_cut = {
		'1d': 12000,
		'3d': 3000
	}[dim]

	total_time = {
		'1d': 0.1,
		'3d': 1.0
	}[dim]

	# Prepare constants and grid
	constants = Constants(double=env.supportsDouble(),
		e_cut=(e_cut if use_cutoff else None),
		**constants_kwds)

	if grid_type == 'uniform':
		if use_cutoff:
			box_size = constants.boxSizeForN(total_N, len(shape))
			min_shape = constants.planeWaveModesForCutoff(box_size)

			# FFT supports 2**n dimensions only
			shape = tuple([2 ** (log2(x - 1) + 1) for x in min_shape])

			if use_big_grid:
				shape = tuple([2 * x for x in shape])

		grid = UniformGrid.forN(env, constants, total_N, shape)
	elif grid_type == 'harmonic':
		if use_cutoff:
			shape = constants.harmonicModesForCutoff(len(shape))
			if use_big_grid:
				shape = tuple([x + 3 for x in shape])

		grid = HarmonicGrid(env, constants, shape)

	if grid_type == 'uniform':
		gs = SplitStepGroundState(env, constants, grid, dt=ss_dt)
	elif grid_type == 'harmonic':
		gs = RK5HarmonicGroundState(env, constants, grid, eps=1e-7)

	if grid_type == 'harmonic':
		evolution = RK5HarmonicEvolution(env, constants, grid,
			atol_coeff=1e-3, eps=1e-6, Nscale=total_N)
	elif prop_type == 'split-step':
		evolution = SplitStepEvolution(env, constants, grid, dt=ss_dt)
	elif prop_type == 'rk5':
		evolution = RK5IPEvolution(env, constants, grid, Nscale=total_N,
			atol_coeff=1e-3, eps=1e-6)

	pulse = Pulse(env, constants, grid, f_detuning=41, f_rabi=350)

	a = AxialProjectionCollector(env, constants, grid, pulse=pulse)
	p = ParticleNumberCollector(env, constants, grid, pulse=pulse)
	v = VisibilityCollector(env, constants, grid)

	# experiment
	psi = gs.create((total_N, 0))

	pulse.apply(psi, theta=0.5 * numpy.pi)

	t1 = time.time()
	evolution.run(psi, total_time, callbacks=[a, p, v], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()

	times, heightmap = a.getData()

	# check that the final state is still projected
	psi.toMSpace()
	mode_data = env.fromDevice(psi.data)
	mask = numpy.tile(buildProjectorMask(constants, grid),
		(psi.components, 1) + (1,) * grid.dim)
	masked_mode_data = mode_data * (1.0 - mask)
	assert masked_mode_data.max() < 1e-6 * mode_data.max()

	res = HeightmapPlot(
		HeightmapData("test", heightmap,
			xmin=0, xmax=total_time * 1e3,
			ymin=grid.z[0] * 1e6,
			ymax=grid.z[-1] * 1e6,
			zmin=-1, zmax=1,
			xname="T (ms)", yname="z ($\\mu$m)", zname="Spin projection")
	)

	times, Ns, Ntotals = p.getData()
	times, vis = v.getData()

	print ("  Shape: {shape}, final N: {N} ({Ntotal}), V: {vis}\n" +
		"  Time spent: {t} s").format(
		shape=shape, N=Ns[:,-1], Ntotal=Ntotals[-1], vis=vis[-1], t=t2-t1)

	return res


if __name__ == '__main__':

	prefix = 'cutoff_'

	tests = (
		('harmonic', 'uniform',), # grid type
		('split-step', 'rk5',), # propagation type
		(False, True,), # cutoff usage
		(False, True,), # use big grid
	)

	for dim in ('1d', '3d',):
		print "\n*** {dim} ***\n".format(dim=dim)

		for grid_type, prop_type, use_cutoff, use_big_grid in itertools.product(*tests):

			if grid_type == 'harmonic' and prop_type == 'split-step':
				continue

			if not use_cutoff and use_big_grid:
				continue

			print "* Testing", ", ".join((
				grid_type, prop_type,
				("cutoff" if use_cutoff else "no cutoff"),
				(("big grid" if use_big_grid else "minimal grid") if use_cutoff else '')))

			testCutoff(dim, grid_type, prop_type, use_cutoff, use_big_grid).save(
				prefix + dim + '_' + grid_type + '_' + prop_type +
				('_cutoff' + ('_biggrid' if use_big_grid else '_mingrid')
					if use_cutoff else '_nocutoff') + '.pdf')
