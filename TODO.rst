--------------------------------
Features/bugfixes for the future
--------------------------------

- Execute callbacks at exact times, not approximate
- Different potentials for each component (static)
- Parse keywords to Constants constructor and extract values/orders of loss coefficients
- Separate tests and examples, add missing examples
- Add reference documentation
- Move helpers folder to a separate project
- Generate "restricted" ground state, i.e. containing only modes below cutoff.
  Possible variants:
  * Build ground state for full basis, remove high-energy modes and rescale remaining ones
  * Build ground state for restricted basis from scratch
  * Build ground state for larger N so that after dropping high-energy modes remaining ones contain exactly required N (this will involve changes in scaling part of ._create())
- Try to build Wigner function (and see if it is negative at some points)
- Remove temporary workaround for datahelpers from beclab/init (try to load matplotlib and
  if succeeded define XYPlot/HeightmapPlot; or just require matplotlib strictly)
- Implement more advanced integration methods (see split-step papers and Num. Rec.)
- Add Peter's tests (with corresponding changes to beclab)
- In tests where CPU and GPU are compared check that their results are identical (for doubles)
