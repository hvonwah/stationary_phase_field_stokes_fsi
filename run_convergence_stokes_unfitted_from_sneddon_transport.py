# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import TaskManager, SetNumThreads, \
    sqrt, pi, CF, x, y, sin, Grad, VTKOutput
from sneddon import *
from unfitted_stokes import *
from xfem import InterpolateToP1
import pickle
from hashlib import blake2b
import argparse

SetNumThreads(12)


parser = argparse.ArgumentParser()
parser.add_argument('-h0', '--meshsize', type=float, default=0.02,
                    help='Mesh size on level 0, default=0.02')
parser.add_argument('-Lx', '--levels', type=int, default=6,
                    help='Number of refinment levels, default=6')
parser.add_argument('-dt0', '--timestep', type=float, default=0.1,
                    help='Timestep for levelset trabsport')
parser.add_argument('-vtk', '--vtk_output', action='store_true', default=False,
                    help='Write VTK output of solution')
args = parser.parse_args()
options = vars(args)


# ----------------------------------- DATA ------------------------------------
h0 = options['meshsize']
Lx = options['levels']
dt0 = options['timestep']

inverse = 'pardiso'
compile_flag = False
wait_compile = False
vtk_flag = options['vtk_output']

# Sneddon Data
data = {'l0': 0.2, 'G_c': 5e2, 'E': 100000, 'nu_s': 0.35, 'p': 4.5e3}

xmin, xmax = 1.7, 2.3
n_cod0 = 20
n_cod_factor = 2

# Stokes Data
nu = 1e-4
a, b = data['l0'], cod_func(0, data) / 2
r2 = ((x - 2) / a)**2 + ((y - 2) / b)**2
r = sqrt(r2)
phi = sin(pi * r2 / 2)
levelset_ex = r2 - 1

u_ex = CF((phi.Diff(y), -phi.Diff(x)))
p_ex = sin(pi * r2 / 2) - 2 / pi
u_ex_grad = CF((u_ex[0].Diff(x), u_ex[0].Diff(y),
                u_ex[1].Diff(x), u_ex[1].Diff(y)), dims=(2, 2))
f1 = - nu * u_ex[0].Diff(x).Diff(x) - nu * u_ex[0].Diff(y).Diff(y)
f1 += p_ex.Diff(x)
f2 = - nu * u_ex[1].Diff(x).Diff(x) - nu * u_ex[1].Diff(y).Diff(y)
f2 += p_ex.Diff(y)
force = CF((f1, f2))

# Output files
_hash = hash((h0, data['l0'], data['G_c'], data['E'], data['nu_s'], data['p']))
_hash = _hash.to_bytes(_hash.bit_length() // 8 + 1, 'little', signed=True)
out_pars = f'{blake2b(_hash, digest_size=6).hexdigest()}'
out_sneddon_file = 'output/sneddon_' + out_pars
out_tcv_file = out_sneddon_file + '_tcv_lset_trans'
out_stokes_file = 'output/stokes_lset_transport_' + out_pars


# ------------------------ CHECK FOR AVAILABLE RESULTS ------------------------
tcv_lset = []
out_stokes = {'l2v': [], 'h1v': [], 'l2p': []}


# ------------------------- SNEDDON CONVERGENCE STUDY -------------------------
with TaskManager():
    for lvl in range(Lx):

        with open(out_sneddon_file + f'_lvl{lvl}.dat', 'rb') as fid:
            gfu_pf = pickle.load(fid)

        lsetp1 = GridFunction(H1(gfu_pf.components[1].space.mesh, order=1))
        gf_u, gf_phi = gfu_pf.components
        InterpolateToP1(gf_phi - ((1 + sqrt(5)) / 2 - 1), lsetp1)

        lsetp1 = make_levelset_from_transport(gfu=gfu_pf, lsetp1=lsetp1,
                                              dt=dt0 * 0.5**lvl, order_def=2,
                                              bc_dir='out', inverse=inverse)

        tcv_lset.append(tcv_from_eulerian_levelset(lsetp1))

        with open(out_tcv_file + '.dat', 'wb') as fid:
            pickle.dump(tcv_lset, fid)

        gfu_stokes = solve_unfitted_stokes(
            lsetp1=lsetp1, f_vol=force, nu=nu, compile_flag=compile_flag,
            wait_compile=wait_compile, inverse=inverse)
        vel, pre, lam = gfu_stokes.components

        out_stokes['l2v'].append(L2Norm(lsetp1, vel - u_ex))
        out_stokes['h1v'].append(L2Norm(lsetp1, Grad(vel) - u_ex_grad))
        out_stokes['l2p'].append(L2Norm(lsetp1, pre - p_ex))

        with open(out_stokes_file + '.dat', 'wb') as fid:
            pickle.dump(out_stokes, fid)

        if vtk_flag:
            vtk = VTKOutput(ma=lsetp1.space.mesh, coefs=[lsetp1, vel, pre],
                            names=['lset', 'vel', 'pre'],
                            filename=out_stokes_file + f'_lvl{lvl}',
                            floatsize='single')
            vtk.Do()

        del gfu_pf, gfu_stokes, lsetp1


# ---------------------------- POSTPROCESS RESULTS ----------------------------
with open(out_stokes_file + '.txt', 'w') as fid:
    fid.write('lvl l2v h1v l2p\n')
    for lvl in range(Lx):
        fid.write(f'{lvl}')
        for key in ['l2v', 'h1v', 'l2p']:
            fid.write(f' {out_stokes[key][lvl]:6.4e}')
        fid.write('\n')

with open(out_tcv_file + '.txt', 'w') as fid:
    fid.write('lvl tcv err\n')
    for lvl, tcv in enumerate(tcv_lset):
        fid.write(f'{lvl} {tcv:8.6e} {abs(tcv - sneddon_vol(data)):5.3e}\n')
