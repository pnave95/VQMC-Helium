# VQMC-Helium
Notes on and implementation of variational quantum monte carlo (VQMC) applied to Helium.

To run a search over an Na x Nb grid on the parameter space rectangle alpha in [amin, amax], beta in [bmin, bmax], using an M sample chain for each grid point Monte Carlo and using delta as the proposal move parameter, call the function

FineGridSearch(amin, amax, Na, bmin, bmax, Nb, M, delta)

in the file 'naive_vqmc.py'

--------------------------------
Other functions exist in the various files to run multiple chains for a single pair (alpha, beta) of parameter values so that convergence can be visualized directly via graphs.
