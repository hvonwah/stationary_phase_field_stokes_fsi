# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import SetNumThreads, TaskManager, VTKOutput
from sneddon import *
import pickle
from hashlib import blake2b
from os.path import exists
import argparse

SetNumThreads(12)


parser = argparse.ArgumentParser()
parser.add_argument('-h0', '--meshsize', type=float, default=0.01,
                    help='Mesh size on level 0, default=0.01')
parser.add_argument('-Lx', '--levels', type=int, default=5,
                    help='Number of refinment levels, default=6')
parser.add_argument('-vtk', '--vtk_output', action='store_true', default=False,
                    help='Write VTK output of solution')
args = parser.parse_args()
options = vars(args)


# ----------------------------------- DATA ------------------------------------
h0 = options['meshsize']
Lx = options['levels']

inverse = 'pardiso'
compile_flag = True
wait_compile = True
vtk_flag = options['vtk_output']

# Sneddon Data
data = {'l0': 0.1, 'G_c': 5e2, 'E': 50000, 'nu_s': 0.35, 'p': 1e4}


# Output files
_hash = hash((h0, data['l0'], data['G_c'], data['E'], data['nu_s'], data['p']))
_hash = _hash.to_bytes(_hash.bit_length() // 8 + 1, 'little', signed=True)
out_pars = f'{blake2b(_hash, digest_size=6).hexdigest()}'
out_sneddon_file = 'output/sneddon_T_' + out_pars


# ------------------------- SNEDDON CONVERGENCE STUDY -------------------------
with TaskManager():
    for lvl in range(Lx):

        pickle_sned = out_sneddon_file + f'_lvl{lvl}.dat'

        if exists(pickle_sned):
            with open(pickle_sned, 'rb') as fid:
                gfu_pf = pickle.load(fid)
        else:
            hmax = h0 * 0.5**lvl
            mesh_factor = 100
            order = 1
            eps = 0.5 * sqrt(hmax)
            gamma = 100 * hmax**-2
            kappa = 1e-10

            gfu_pf = sneddon_T_stationary(
                data, hmax=hmax, mesh_factor=mesh_factor, eps=eps, gamma=gamma,
                order=order, kappa=kappa, newton_solver=inverse,
                compile_flag=compile_flag, wait_compile=wait_compile)

            with open(pickle_sned, 'wb') as fid:
                pickle.dump(gfu_pf, fid)

            if vtk_flag:
                vtk = VTKOutput(ma=gfu_pf.components[0].space.mesh,
                                coefs=[*gfu_pf.components], names=['u', 'phi'],
                                filename=pickle_sned[:-4], floatsize='single')
                vtk.Do()
