#!/bin/bash

mkdir -p output

/usr/bin/time -v python3 run_convergence_sneddon.py --meshsize 0.02 --levels 6
/usr/bin/time -v python3 run_convergence_stokes_fitted_from_sneddon.py --meshsize 0.02 --levels 6 --order 2
/usr/bin/time -v python3 run_convergence_stokes_unfitted_from_sneddon_explicit.py --meshsize 0.02 --levels 6
/usr/bin/time -v python3 run_convergence_stokes_unfitted_from_sneddon_transport.py --meshsize 0.02 --levels 6 --timestep 0.1
/usr/bin/time -v python3 run_convergence_stationary_fsi.py --meshsize 0.02 --levels 5 --order 2
/usr/bin/time -v python3 run_convergence_sneddonT.py         --meshsize 0.01 --levels 6
/usr/bin/time -v python3 run_convergence_stationary_fsi_T.py --meshsize 0.01 --levels 6 --order 2
