# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.occ import *
from ngsolve import *
from sneddon import *
from stationary_stokes_fsi import stationary_stokes_fsi
import pickle
from hashlib import blake2b
import argparse

ngsglobals.msg_level = 0
SetHeapSize(10000000)
SetNumThreads(4)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-h0', '--meshsize', type=float, default=0.01,
                    help='Mesh size on level 0')
parser.add_argument('-Lx', '--levels', type=int, default=6,
                    help='Number of refinment levels')
parser.add_argument('-o', '--order', type=int, default=2,
                    help='Order for velocity space')
parser.add_argument('-vtk', '--vtk_output', action='store_true',
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
data = {'l0': 0.1, 'G_c': 5e2, 'E': 50000, 'nu_s': 0.35, 'p': 1e4}

n_cod0 = 20
n_cod_factor = 1

mus = data['E'] / (2 * (1 + data['nu_s']))
ls = data['E'] * data['nu_s'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))

# FSI Data
c1, c2 = 0.0001, 5000
x0 = (2.098, 2.002)
rhs_data_fluid = CF((0, c1 * exp(-c2 * ((x - x0[0])**2 + (y - x0[1])**2))))

data_mat = {'rhof': 1e3, 'nuf': 0.1, 'mus': mus, 'lams': ls}

x_eval = (2.05, 2.025)
# Output files
_hash = hash((h0, data['l0'], data['G_c'], data['E'], data['nu_s'], data['p']))
_hash = _hash.to_bytes(_hash.bit_length() // 8 + 1, 'little', signed=True)
out_pars = f'{blake2b(_hash, digest_size=6).hexdigest()}'
out_sneddon_file = 'output/sneddon_T_' + out_pars
out_fsi_file = 'output/fsi_' + out_pars


# ------------------------- SNEDDON CONVERGENCE STUDY -------------------------
with open(out_fsi_file + '.txt', 'w') as fid:
    fid.write('lvl drag lift def_x, def_y\n')

with TaskManager():
    for lvl in range(Lx):
        print(f'lvl = {lvl}')

        with open(out_sneddon_file + f'_lvl{lvl}.dat', 'rb') as fid:
            gfu_pf = pickle.load(fid)

        hmax = h0 * 0.5**lvl

        # Construct geometry from crack opening displacements
        n_cod = int(n_cod0 * n_cod_factor**lvl)
        l0 = data['l0']
        cod_offset = max(2 * hmax, 1e-3)

        lines_x = [2 + l0 - cod_offset - i * 2 * (l0 - cod_offset) / (n_cod - 1)
                   for i in range(n_cod)][2:]
        cod = cod_from_phase_field(gfu_pf, lines_x)

        bnd_pts1 = [(p[0], 2 + p[1] / 2) for p in cod] + [(2 - l0, 2)]
        bnd_pts1 += [(p[0], 2 - p[1] / 2) for p in reversed(cod)]

        n_cod = int(ceil(n_cod / 2))
        cod_offset = max(2 * hmax, 6e-3)
        lines_y = [2 - l0 + cod_offset + i * (l0 - 3 * cod_offset) / (n_cod - 1)
                   for i in range(n_cod)]
        lines_y.reverse()

        with TaskManager():
            cod = cod_from_phase_field(gfu_pf, lines_y, True)

        bnd_pts2 = [(2 + l0 - p[1] / 2, p[0]) for p in cod]
        bnd_pts2 += [(2 + l0, 2 - l0)]
        bnd_pts2 += [(2 + l0 + p[1] / 2, p[0]) for p in reversed(cod)]

        lines_y = [2 + l0 - cod_offset - i * (l0 - 3 * cod_offset) / (n_cod - 1)
                   for i in range(n_cod)]
        lines_y.reverse()
        with TaskManager():
            cod = cod_from_phase_field(gfu_pf, lines_y, True)

        bnd_pts2 += [(2 + l0 + p[1] / 2, p[0]) for p in cod]
        bnd_pts2 += [(2 + l0, 2 + l0)]
        bnd_pts2 += [(2 + l0 - p[1] / 2, p[0]) for p in reversed(cod)]

        geo1 = SplineApproximation(bnd_pts1, deg_min=2, deg_max=2,
                                   continuity=ShapeContinuity(2), tol=1e-9)
        geo2 = SplineApproximation(bnd_pts1[-1:] + bnd_pts2 + bnd_pts1[:1],
                                   deg_min=2, deg_max=2,
                                   continuity=ShapeContinuity(2), tol=1e-9)

        fluid = Face(Wire([geo1.Wire(), geo2.Wire()]))
        fluid.faces.name = 'fluid'
        fluid.faces.maxh = hmax * 0.5
        fluid.edges.name = 'interface'

        base = MoveTo(0, 0).Rectangle(4, 4).Face()
        base.edges.name = 'out'
        base -= fluid
        base.faces.name = 'solid'
        base.faces.maxh = 200 * hmax

        geo = Glue([base, fluid])

        mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(curvaturesafety=0.1))

        if order > 1:
            mesh.Curve(order)

        # Solve Stationary Stokes FSI Problem
        gfu, drag, lift = stationary_stokes_fsi(mesh, order, data_mat,
                                                rhs_data_fluid)
        vel, pre, deform = gfu.components
        d0, d1 = deform(mesh(*x_eval))

        with open(out_fsi_file + '.txt', 'a') as fid:
            fid.write(f'{lvl} {drag:7.5e} {lift:7.5e} {d0:7.5e} {d1:7.5e}\n')

        if vtk_flag:
            pre_cf = CoefficientFunction([pre if mat == 'fluid' else CF(float('NaN'))
                                          for mat in mesh.GetMaterials()])
            vel_cf = CoefficientFunction([vel if mat == 'fluid' else CF((float('NaN'), float('NaN')))
                                          for mat in mesh.GetMaterials()])
            vtk = VTKOutput(ma=mesh, coefs=[vel_cf, pre_cf, deform],
                            names=['vel', 'pre', 'deform'],
                            filename=out_fsi_file + f'_lvl{lvl}',
                            floatsize='single')
            vtk.Do()
