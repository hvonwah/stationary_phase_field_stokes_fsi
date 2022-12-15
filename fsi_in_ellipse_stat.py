# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions
import argparse

SetNumThreads(12)

parser = argparse.ArgumentParser()
parser.add_argument('-hmax', '--mesh_size', type=float, default=0.12)
parser.add_argument('-o', '--order', type=int, default=2)

args = parser.parse_args()
options = vars(args)


# -------------------------------- PARAMETERS ---------------------------------
hmax = options['mesh_size']
order = options['order']

harmonic_extension = True
alpha_u = 1e-14

compile_flag = False
newton_print = False
newton_damp = 1
newton_tol = 1e-15
inverse = "pardiso"


# ----------------------------------- DATA ------------------------------------
rhof = 1e3
nuf = 0.1
rhos = 1e3

data = {'l0': 0.2, 'G_c': 1, 'E': 100000, 'nu_s': 0.35, 'p': 4.5e3}

mus = data['E'] / (2 * (1 + data['nu_s']))
ls = data['E'] * data['nu_s'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))

e_a = data['l0']
e_b = 2 * (1 - data['nu_s']**2) * data['l0'] * data['p'] / data['E']

c1, c2 = 0.0001, 1000
x0 = (2 + e_a * 1 / 4, 2 + e_b * 2 / 3)
rhs_data_fluid = CF((0, c1 * exp(-c2 * ((x - x0[0])**2 + (y - x0[1])**2))))


# ----------------------------------- MESH ------------------------------------
geo = SplineGeometry()
geo.AddRectangle((0, 0), (4, 4), bcs=['out', 'out', 'out', 'out'],
                 leftdomain=1, rightdomain=0)
pnts = [(2, 2 - e_b), (2 + e_a, 2 - e_b), (2 + e_a, 2), (2 + e_a, 2 + e_b),
        (2, 2 + e_b), (2 - e_a, 2 + e_b), (2 - e_a, 2), (2 - e_a, 2 - e_b)]
pts = [geo.AppendPoint(*p) for p in pnts]
for p1, p2, p3 in [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0)]:
    geo.Append(['spline3', pts[p1], pts[p2], pts[p3]], bc='interface',
               leftdomain=2, rightdomain=1, maxh=hmax / 50)
geo.SetMaterial(1, 'solid')
geo.SetMaterial(2, 'fluid')
geo.SetDomainMaxH(2, hmax / 50)

mesh = Mesh(geo.GenerateMesh(maxh=hmax, curvaturesafety=0.01))
mesh.Curve(order)


# --------------------------- FINITE ELEMENT SPACE ----------------------------
V = VectorH1(mesh, order=order, dirichlet="out")
Q = H1(mesh, order=order - 1, definedon="fluid")
D = VectorH1(mesh, order=order, dirichlet="out")
X = FESpace([V, Q, D])
Y = FESpace([V, Q])
(u, p, d), (v, q, w) = X.TnT()

gfu = GridFunction(X)
velocity, pressure, deformation = gfu.components


# ----------------------------- (BI)LINEAR FORMS ------------------------------
Id2 = Id(2)
F = Grad(d) + Id2
C = F.trans * F
E = 0.5 * (Grad(d) + Grad(d).trans)
J = Det(F)
Finv = Inv(F)
FinvT = Finv.trans

# For Stokes problem
stokes = nuf * rhof * InnerProduct(grad(u), grad(v))
stokes += - div(u) * q - div(v) * p - 1e-9 * p * q

diff_fl = rhof * nuf * InnerProduct(J * grad(u) * Finv * FinvT, grad(v))
# conv_fl = rhof * InnerProduct(J * (grad(u) * Finv) * u, v)
pres_fl = -J * (Trace(grad(v) * Finv) * p + Trace(grad(u) * Finv) * q)
pres_fl += - J * 1e-9 * p * q

rhs_fl = - InnerProduct(rhof * J * rhs_data_fluid, v)

mass_sol = - InnerProduct(u, w)
el_sol = InnerProduct(J * (2 * mus * E + ls * Trace(E) * Id2) * FinvT, grad(v))

if harmonic_extension:
    extension = alpha_u * InnerProduct(Grad(d), Grad(d))
else:
    gfdist = GridFunction(H1(mesh, order=1, dirichlet="inlet|wall|outcyl|outlet"))
    gfdist.Set(1, definedon=mesh.Boundaries("interface"))

    def NeoHookExt(C, mu=1, lam=1):
        return 0.5 * mu * (Trace(C - Id2) + 2 * mu / lam * Det(C)**(-lam / 2 / mu) - 1)

    extension = 1 / (1 - gfdist + 1e-2) * 1e-8 * NeoHookExt(C)


# -------------------------------- INTEGRATORS --------------------------------
comp_opt = {'realcompile': compile_flag, 'wait': True}
dFL, dSL = dx("fluid"), dx("solid")

a = BilinearForm(X, symmetric=False, condense=True)
a += (diff_fl + pres_fl + rhs_fl).Compile(**comp_opt) * dFL
a += (mass_sol + el_sol).Compile(**comp_opt) * dSL
a += Variation(extension.Compile(**comp_opt) * dFL)

a_stokes = BilinearForm(Y, symmetric=True, check_unused=False)
a_stokes += stokes * dFL

f_stokes = LinearForm(Y)
f_stokes += InnerProduct(rhs_data_fluid, v) * dFL


# ------------------------------- VISUALISATION -------------------------------
Draw(CoefficientFunction([velocity if mat == "fluid" else None
                          for mat in mesh.GetMaterials()]), mesh, "velocity")
Draw(pressure, mesh, "pressure")
Draw(deformation, mesh, "deformation")
visoptions.scalfunction = "velocity:0"
visoptions.vecfunction = "deformation"
SetVisualization(deformation=True)


# -------------------------- FUNCTIONAL COMPUTATION ---------------------------
gfu_drag, gfu_lift = GridFunction(X), GridFunction(X)
gfu_drag.components[0].Set(CF((1, 0)), definedon=mesh.Boundaries('interface'))
gfu_lift.components[0].Set(CF((0, 1)), definedon=mesh.Boundaries('interface'))
res = gfu.vec.CreateVector()


# ------------------------------- SOLVE PROBLEM -------------------------------
bts = Y.FreeDofs() & ~Y.GetDofs(mesh.Materials("solid"))
bts &= ~Y.GetDofs(mesh.Boundaries("out|interface"))
bts[Y.Range(1)] = True

rstokes = GridFunction(Y)

with TaskManager():
    a_stokes.Assemble()
    f_stokes.Assemble()
    invstoke = a_stokes.mat.Inverse(bts, inverse="sparsecholesky")

    rstokes.vec.data = invstoke * f_stokes.vec

    gfu.components[0].vec.data = rstokes.components[0].vec
    gfu.components[1].vec.data = rstokes.components[1].vec

Redraw(blocking=True)

with TaskManager():
    solvers.Newton(a, gfu, maxit=10, inverse=inverse, maxerr=newton_tol,
                   dampfactor=newton_damp, printing=newton_print)
    Redraw(blocking=False)
    a.Apply(gfu.vec, res)
    drag = - InnerProduct(res, gfu_drag.vec)
    lift = - InnerProduct(res, gfu_lift.vec)
    d0, d1 = deformation(mesh(2 + e_a / 2, 2 + e_b))
    print(f'{order} {hmax:4.2e} {drag: 10.7e} {lift: 10.7e} ', end='')
    print(f'{d0: 10.7e} {d1: 10.7e}')
