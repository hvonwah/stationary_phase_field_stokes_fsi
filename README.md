[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7443025.svg)](https://doi.org/10.5281/zenodo.7443025)


This repository contains the reproduction scripts and resulting data for the paper "A high-precision framework for phase-field fracture interface reconstructions with application to Stokes fluid-filled fracture surrounded by an elastic medium" by H. v. Wahl and T. Wick.

# Files
```
+- README.md                                                    // This file
+- LICENSE                                                      // The licence file
+- install.txt                                                  // Installation help
+- run_all.bash                                                 // Script to run all examples
+- run_convergence_sneddon.py                                   // Convergence study for Sneddon's test
+- run_convergence_stokes_unfitted_from_sneddon_transport.py    // Convergence study for level set transport reconstruction
+- run_convergence_stokes_unfitted_from_sneddon_explicit.py     // Convergence study for explicit level set reconstruction
+- run_convergence_stokes_fitted_from_sneddon.py                // Convergence study from fitted interface reconstruction
+- run_convergence_stationary_fsi.py                            // Convergence Study for FSI in opening crack
+- sneddon.py                                                   // Implementation for Sneddon's test
+- stokes.py                                                    // Implementation of a fitted Stokes solver
+- unfitted_stokes.py                                           // Implementation of an unfitted Stokes solver
+- stationary_stokes_fsi.py                                     // Implementation of a stationary Stokes FSI solver
+- run_fsi_in_ellipse.bash                                      // Script for FSI convergence study in exact domain
+- fsi_in_ellipse_stat.py                                       // Implementation of Stokes FSI in apocalyptically exact ellipse fluid domain
+- output/*                                                     // The raw text files produced by the computations 
```

# Installation

See the instructions in `install.txt`

# How to reproduce
The scripts to reproduce the computational results are located in the base folder. The resulting data is located in the `output` directory.

The individual convergence studies presented are computed using the `run_convergence_*` scripts. All the examples presented in our paper are reproduced by running the bash script `run_all.bash`.

By default, the direct solver `pardiso` is used to solve the linear systems resulting from the discretisation. If this is not available, this may be replaced with `umfpack` in the `DATA` block of each convergence study script.

The `run_convergence_sneddon.py` pickles the finite element solution to the phase-field problem, and the remaining convergence studies reuse this solution if available. These files are not stored in this repository.

# Reference fluid-structure interaction simulation
The reverence values for the stationary fluid-structure interaction problem were determined by solving the problem on the exact domain resulting asymptotically from Sneddon's test. This is implemented in `fsi_in_ellipse_stat.py`, and the convergence study is run by running the script `run_fsi_in_ellipse.bash`.