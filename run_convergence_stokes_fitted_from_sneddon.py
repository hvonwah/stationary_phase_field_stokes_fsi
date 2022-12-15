# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from sneddon import *
from stokes import *
import pickle
from hashlib import blake2b
import argparse

SetNumThreads(12)


parser = argparse.ArgumentParser()
parser.add_argument('-h0', '--meshsize', type=float, default=0.02,
                    help='Mesh size on level 0, default=0.02')
parser.add_argument('-Lx', '--levels', type=int, default=6,
                    help='Number of refinment levels, default=6')
parser.add_argument('-o', '--order', type=int, default=2,
                    help='Order for Stokes velocity space, default=2')
parser.add_argument('-vtk', '--vtk_output', action='store_true', default=False,
                    help='Write VTK output of solution')
args = parser.parse_args()
options = vars(args)


# ----------------------------------- DATA ------------------------------------
h0 = options['meshsize']
Lx = options['levels']
order_stokes = options['order']

condense_flag = False
inverse = 'pardiso'
compile_flag = False
wait_compile = False
vtk_flag = options['vtk_output']

# Sneddon Data
data = {'l0': 0.2, 'G_c': 5e2, 'E': 100000, 'nu_s': 0.35, 'p': 4.5e3}

xmin, xmax = 1.7, 2.3
n_cod0 = 10
n_cod_factor = 1

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
out_tcv_file = out_sneddon_file + '_tcv_spline_mesh'
out_stokes_file = 'output/stokes_spline_' + out_pars


# ------------------------ CHECK FOR AVAILABLE RESULTS ------------------------
tcv_dom = []
out_stokes = {'l2v': [], 'h1v': [], 'l2p': []}


# ------------------------- SNEDDON CONVERGENCE STUDY -------------------------
with TaskManager():
    for lvl in range(Lx):

        with open(out_sneddon_file + f'_lvl{lvl}.dat', 'rb') as fid:
            gfu_pf = pickle.load(fid)

        n_cod = int(n_cod0 * n_cod_factor**lvl)
        lines = [xmin + i * (xmax - xmin) / (n_cod - 1) for i in range(n_cod)]
        cod_pf = cod_from_phase_field(gfu_pf, lines)

        points1 = [(2 - data['l0'], 2)] + [(p, 2 + h / 2) for p, h in cod_pf
                                           if abs(p - 2) < data['l0'] - 1e-10]
        points2 = [(2 + data['l0'], 2)]
        points2 = points2 + [(p, 4 - h) for p, h in reversed(points1[1:])]

        mesh = make_mesh_from_points(points1 + points2, hmax=h0 * 2**-lvl,
                                     curvaturesafety=0.01)

        if order_stokes > 1:
            mesh.Curve(order_stokes)
        tcv_dom.append(Integrate(CF(1) * dx, mesh))

        with open(out_tcv_file + '.dat', 'wb') as fid:
            pickle.dump(tcv_dom, fid)

        vel, pre = solve_stokes(
            mesh=mesh, order=order_stokes, dirichlet='crack', f_vol=force,
            nu=nu, compile_flag=compile_flag, wait_compile=wait_compile,
            inverse=inverse, condense=condense_flag, lagr=True)

        out_stokes['l2v'].append(L2Norm(mesh, vel - u_ex))
        out_stokes['h1v'].append(L2Norm(mesh, Grad(vel) - u_ex_grad))
        out_stokes['l2p'].append(L2Norm(mesh, pre - p_ex))

        with open(out_stokes_file + '.dat', 'wb') as fid:
            pickle.dump(out_stokes, fid)

        if vtk_flag:
            vtk = VTKOutput(ma=mesh, coefs=[vel, pre, u_ex, p_ex],
                            names=['vel', 'pre', 'uex', 'pex'],
                            filename=out_stokes_file + f'_lvl{lvl}',
                            floatsize='single')
            vtk.Do()

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
    for lvl, tcv in enumerate(tcv_dom):
        fid.write(f'{lvl} {tcv:8.6e} {abs(tcv - sneddon_vol(data)):5.3e}\n')
