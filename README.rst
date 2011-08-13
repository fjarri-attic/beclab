~~~~~~~~~~~~~~~~~~~~~
Trapped BEC simulator
~~~~~~~~~~~~~~~~~~~~~

============
Introduction
============

This module is used for simulating the behavior of multi-component BEC in a magnetic trap.
Currently it supports common GPEs and Wigner representation (with plane wave or harmonic wave decomposition).
Simulation results can be saved as JSON files or several types of graphs; see examples for details.

===============
Release history
===============

-------------------------
0.2.0 (under development)
-------------------------

* Added harmonic decomposition support
* Separated global constants, grid parameters and local algorithm parameters
* Removed hardcoding of 2 components (except several places where it does not affect the whole program)
* Added RK5IP and RK5Harmonic ground states
* Added RK5IP evolution
* Added some tests (mostly functionality and coverage)
* Executing callbacks at exact times, not approximate
* Support for "restricted" ground states, i.e. containing only modes below cutoff

TODO:
* Add support for different potentials for each component (static)
* Separate tests and examples
* Add performance tests, profile evolution and try to optimise it

-----
0.1.3
-----

This was the working version. Changes were not recorded before this.
