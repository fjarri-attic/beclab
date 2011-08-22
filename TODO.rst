--------------------------------
Features/bugfixes for the future
--------------------------------

- Parse keywords to Constants constructor and extract values/orders of loss coefficients
- Add reference documentation
- Move helpers folder to a separate project
- Try to build Wigner function (and see if it is negative at some points)
- Remove temporary workaround for datahelpers from beclab/init (try to load matplotlib and
  if succeeded define XYPlot/HeightmapPlot; or just require matplotlib strictly)
- Implement more advanced integration methods (see split-step papers and Num. Rec.)
- Add Peter's tests (with corresponding changes to beclab)
- In tests where CPU and GPU are compared add an assertion which checks that their
  results are identical (for doubles)
- Implement more accurate propagation schemes for ground states (from Bao et al., 2004)
