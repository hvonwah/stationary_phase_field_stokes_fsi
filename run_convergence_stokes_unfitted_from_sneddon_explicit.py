# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import TaskManager, SetNumThreads, \
    sqrt, pi, CF, x, y, sin, Grad, VTKOutput
from sneddon import *
from unfitted_stokes import *
import pickle
from hashlib import blake2b
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
compile_flag = False
wait_compile = False
vtk_flag = options['vtk_output']

# Sneddon Data
data = {'l0': 0.2, 'G_c': 5e2, 'E': 100000, 'nu_s': 0.35, 'p': 4.5e3}

xmin, xmax = 1.7, 2.3
n_cod0 = 12
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
out_tcv_file = out_sneddon_file + '_tcv_lset_expl'
out_stokes_file = 'output/stokes_lset_explicit_' + out_pars


# ------------------------ CHECK FOR AVAILABLE RESULTS ------------------------
tcv_lset = []
out_stokes = {'l2v': [], 'h1v': [], 'l2p': []}


# ------------------------- SNEDDON CONVERGENCE STUDY -------------------------
with TaskManager():
    for lvl in range(Lx):

        with open(out_sneddon_file + f'_lvl{lvl}.dat', 'rb') as fid:
            gfu_pf = pickle.load(fid)

        n_cod = int(n_cod0 * n_cod_factor**lvl)
        lines = [xmin + i * (xmax - xmin) / (n_cod - 1) for i in range(n_cod)]

        cod = [_c for _c in cod_from_phase_field(gfu_pf, lines)
               if abs(_c[0] - 2) < data['l0'] - 1e-10]
        cod = [(2 - data['l0'], 0)] + cod + [(2 + data['l0'], 0)]

        remove = non_convex_points(cod)
        while len(remove) > 0:
            cod = [cod[i] for i in range(len(cod)) if i not in remove]
            remove = non_convex_points(cod)
        print(f'lvl = {lvl}, {len(cod)} cod used for levelset')

        mesh = Mesh(gfu_pf.components[0].space.mesh.ngmesh.Copy())

        dta = make_multiple_levelset_from_cod(cod, mesh)
        print('dta has length', len(dta))
        tcv_lset.append(tcv_from_eulerian_multiple_levelset(dta))

        with open(out_tcv_file + '.dat', 'wb') as fid:
            pickle.dump(tcv_lset, fid)

        gfu_stokes = solve_unfitted_stokes_mlset(
            mlset_dom=dta, f_vol=force, nu=nu, compile_flag=compile_flag,
            wait_compile=wait_compile, inverse=inverse)
        vel, pre, lam = gfu_stokes.components

        out_stokes['l2v'].append(L2Norm_mlset(dta, vel - u_ex))
        out_stokes['h1v'].append(L2Norm_mlset(dta, Grad(vel) - u_ex_grad))
        out_stokes['l2p'].append(L2Norm_mlset(dta, pre - p_ex))

        with open(out_stokes_file + '.dat', 'wb') as fid:
            pickle.dump(out_stokes, fid)

        if vtk_flag:
            if len(dta.as_list) > 1:
                print(f'lvl={lvl} -- VTK a.t.m. only with a single levelset')
            else:
                vtk = VTKOutput(ma=dta.lsets[0].space.mesh,
                                coefs=[dta.lsets[0], vel, pre],
                                names=['lset', 'vel', 'pre'],
                                filename=out_stokes_file + f'_lvl{lvl}',
                                floatsize='single')
            vtk.Do()

        del gfu_pf, gfu_stokes, dta, mesh


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
