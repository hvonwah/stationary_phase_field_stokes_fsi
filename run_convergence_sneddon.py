# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import SetNumThreads, TaskManager, VTKOutput
from sneddon import *
from stokes import *
import pickle
from hashlib import blake2b
from os.path import exists
import argparse

SetNumThreads(12)


parser = argparse.ArgumentParser()
parser.add_argument('-h0', '--meshsize', type=float, default=0.02,
                    help='Mesh size on level 0, default=0.02')
parser.add_argument('-Lx', '--levels', type=int, default=6,
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
data = {'l0': 0.2, 'G_c': 5e2, 'E': 100000, 'nu_s': 0.35, 'p': 4.5e3}

xmin, xmax = 1.7, 2.3
n_cod0 = 15
n_cod_factor = 1


# Output files
_hash = hash((h0, data['l0'], data['G_c'], data['E'], data['nu_s'], data['p']))
_hash = _hash.to_bytes(_hash.bit_length() // 8 + 1, 'little', signed=True)
out_pars = f'{blake2b(_hash, digest_size=6).hexdigest()}'
out_sneddon_file = 'output/sneddon_' + out_pars


# ------------------------ CHECK FOR AVAILABLE RESULTS ------------------------
out_sneddon = {'cod_pf': [], 'cod_pt': [], 'tcv_pf': []}


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

            gfu_pf = sneddon_stationary(
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

        n_cod = int(n_cod0 * n_cod_factor**lvl)
        lines = [xmin + i * (xmax - xmin) / (n_cod - 1) for i in range(n_cod)]
        out_sneddon['cod_pf'].append(cod_from_phase_field(gfu_pf, lines))
        out_sneddon['cod_pt'].append(cod_from_phase_field_lset(gfu_pf, lines))
        out_sneddon['tcv_pf'].append(tcv_from_phase_field(gfu_pf))

        with open(out_sneddon_file + '_results.dat', 'wb') as fid:
            pickle.dump(out_sneddon, fid)


# ---------------------------- POSTPROCESS RESULTS ----------------------------
# TCV results and errors
with open(out_sneddon_file + '_tcv_pf.txt', 'w') as fid:
    fid.write('lvl tcv err\n')
    for lvl, tcv in enumerate(out_sneddon['tcv_pf']):
        fid.write(f'{lvl} {tcv:8.6e} {abs(tcv - sneddon_vol(data)):5.3e}\n')

# COD results
for lvl in range(Lx):
    with open(out_sneddon_file + f'_cod_lvl{lvl}.txt', 'w') as fid:
        fid.write('pt cod_pf cod_pt\n')
        for (p, cod_pf), (_p, cod_pt) in zip(out_sneddon['cod_pf'][lvl],
                                             out_sneddon['cod_pt'][lvl]):
            fid.write(f'{p - 2:6.4e} {cod_pf:8.6e} {cod_pt:8.6e}\n')


with open(out_sneddon_file + f'_cod_ex.txt', 'w') as fid:
    fid.write('pt cod\n')
    _n = 300
    for p in [xmin + i * (xmax - xmin) / (_n - 1) - 2 for i in range(_n)]:
        fid.write(f'{p:6.4e} {cod_func(p, data):8.6e}\n')


# COD errors
if n_cod_factor == 1:
    with open(out_sneddon_file + f'_cod_conv.txt', 'w') as fid:
        fid.write('lvl')
        for p in lines:
            fid.write(f' err-pf({p - 2:6.4e}) err-pt({p - 2:6.4e})')
        fid.write('\n')

        for lvl in range(Lx):
            fid.write(f'{lvl}')
            for (p, cod_pf), (_p, cod_pt) in zip(out_sneddon['cod_pf'][lvl],
                                                 out_sneddon['cod_pt'][lvl]):
                cod_ex = cod_func(p - 2, data)
                err_pf = abs(cod_pf - cod_ex)
                err_pt = abs(cod_pt - cod_ex)
                fid.write(f' {err_pf:6.4e} {err_pt:6.4e}')
            fid.write('\n')
