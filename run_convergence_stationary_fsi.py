# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from sneddon import *
from stationary_stokes_fsi import stationary_stokes_fsi
import pickle
from hashlib import blake2b
import argparse

SetNumThreads(12)


parser = argparse.ArgumentParser()
parser.add_argument('-h0', '--meshsize', type=float, default=0.02,
                    help='Mesh size on level 0, default=0.02')
parser.add_argument('-Lx', '--levels', type=int, default=5,
                    help='Number of refinment levels, default=5')
parser.add_argument('-o', '--order', type=int, default=2,
                    help='Order for velocity space, default=2')
parser.add_argument('-vtk', '--vtk_output', action='store_true', default=False,
                    help='Write VTK output of solution')
args = parser.parse_args()
options = vars(args)


# ----------------------------------- DATA ------------------------------------
h0 = options['meshsize']
Lx = options['levels']
order = options['order']

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

mus = data['E'] / (2 * (1 + data['nu_s']))
ls = data['E'] * data['nu_s'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))

# FSI Data
c1, c2 = 0.0001, 1000
x0 = (2.05, 2.01053)
rhs_data_fluid = CF((0, c1 * exp(-c2 * ((x - x0[0])**2 + (y - x0[1])**2))))

data_mat = {'rhof': 1e3, 'nuf': 0.1, 'mus': mus, 'lams': ls}

x_eval = (2.1, 2.015795)
# Output files
_hash = hash((h0, data['l0'], data['G_c'], data['E'], data['nu_s'], data['p']))
_hash = _hash.to_bytes(_hash.bit_length() // 8 + 1, 'little', signed=True)
out_pars = f'{blake2b(_hash, digest_size=6).hexdigest()}'
out_sneddon_file = 'output/sneddon_' + out_pars
out_fsi_file = 'output/fsi_' + out_pars


# ------------------------- SNEDDON CONVERGENCE STUDY -------------------------
with open(out_fsi_file + '.txt', 'w') as fid:
    fid.write('lvl drag lift def_x, def_y\n')

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

        mesh = make_fsi_mesh_from_points([(0, 0), (4, 4)], points1 + points2,
                                         hmax=h0 * 2**-(lvl + 2), solid_fac=100,
                                         curvaturesafety=0.3)

        if order > 1:
            mesh.Curve(order)

        gfu, drag, lift = stationary_stokes_fsi(mesh, order, data_mat,
                                                rhs_data_fluid)
        vel, pre, deform = gfu.components
        d0, d1 = deform(mesh(*x_eval))

        with open(out_fsi_file + '.txt', 'a') as fid:
            fid.write(f'{lvl} {drag:7.5e} {lift:7.5e} {d0:7.5e} {d1:7.5e}\n')

        if vtk_flag:
            pre_cf = CoefficientFunction([pre if mat == "fluid" else CF(float('NaN'))
                                          for mat in mesh.GetMaterials()])
            vtk = VTKOutput(ma=mesh, coefs=[vel, pre_cf, deform],
                            names=['vel', 'pre', 'deform'],
                            filename=out_fsi_file + f'_lvl{lvl}',
                            floatsize='single')
            vtk.Do()
