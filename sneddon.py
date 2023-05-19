from netgen.geom2d import SplineGeometry
from netgen.occ import *
from ngsolve import *
from ngsolve.solvers import Newton, PreconditionedRichardson
from xfem import *
from xfem.mlset import *
import warnings
import numpy as np
from more_itertools import pairwise


def cod_func(x, data):
    '''
    Compute the crack opening width at a point x (relative to the
    centre) for Sneddon's problem based on data.

    Parameters
    ----------
    x : float
        Point, relative to the centre of the crack, at which the crack
        opening width is computed.
    data : dict{'l0', 'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.

    Returns
    -------
    crack opening width : float
    '''

    Eprime = data['E'] / (1 - data['nu_s']**2)
    if abs(x) > data['l0']:
        _cod = 0
    else:
        _cod = 4 * data['p'] * data['l0'] / Eprime
        _cod *= sqrt(1 - (x / data['l0'])**2)
    return _cod


def sneddon_vol(data):
    '''
    Compute the volume (tcv) for Sneddon's problem based on data.

    Parameters
    ----------
    data : dict{'l0', 'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.

    Returns
    -------
    total crack volume : float
    '''
    vol_ex = (1 - data['nu_s']**2) * data['l0']**2 * data['p'] / data['E']
    vol_ex *= 2 * pi
    return vol_ex


def phase_field_stationary(data, mesh, bc_dir, eps, gamma, order=1,
                           kappa=1e-10, newton_tol=1e-8, newton_print=False,
                           newton_solver='umfpack', pseudo_time_steps=5,
                           compile_flag=False, wait_compile=False, **kwargs):
    '''
    We use a phase-field approach to compute an opening crack. We then
    construct a level set of the crack geometry by matching the crack
    volume computed from the phase field with the level set crack volume
    using a Newton scheme.

    Parameters
    ----------
    data : dict{'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.
    mesh : ngsolve.Mesh
        Mesh with material 'crack' for phase-field initialisation
    bc_dir : string
        Name of the outer boundary.
    eps : float
        Phase field parameter.
    gamma : float
        Constraint penalty parameter.
    order : float
        Polynomial order of phase field and solid finite element spaces.
    kappa : float
        Regularisation parameter.
    newton_tol : float
        Tolerance for phase field Newton scheme.
    newton_print : boolean
        Print output from phase filed Newton scheme.
    newton_solver : string
        Direct solver to use in Newton scheme.
    pseudo_time_steps : int
        Number of pseudo time step for which we solve the phase-field
        problem.
    compile_flag : boolean
        Strong compile weak form.
    wait_compile : boolean
        Wait for cpp compile to complete.

    Returns
    -------
    gfu : ngsolve.GridFunction
        Resulting finite element function containing the phase-field and
        displacement field.

    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'phase_field_stationary', category=SyntaxWarning)

    G_c = data['G_c']
    E = data['E']
    nu_s = data['nu_s']
    p = data['p']
    mu_s = E / (2 * (1 + nu_s))
    lam_s = nu_s * E / ((1 + nu_s) * (1 - 2 * nu_s))

    # ------------------------- FINITE ELEMENT SPACE --------------------------
    V = VectorH1(mesh, order=order, dirichlet=bc_dir)
    W = H1(mesh, order=order)
    X = V * W
    print(f'The finite element space has {sum(X.FreeDofs())} unknowns')

    (u, phi), (v, psi) = X.TnT()

    gfu, gfu_last = GridFunction(X), GridFunction(X)
    gf_u, gf_phi = gfu.components
    phi_last = gfu_last.components[1]

    def e(U):
        return 1 / 2 * (Grad(U) + Grad(U).trans)

    def sigma(U):
        return 2 * mu_s * e(U) + lam_s * Trace(e(U)) * Id(mesh.dim)

    def pos(x):
        return IfPos(x, x, 0)

    form = ((1 - kappa) * phi_last**2 + kappa) * InnerProduct(sigma(u), e(v))
    form += phi_last**2 * p * div(v)
    form += (1 - kappa) * phi * InnerProduct(sigma(u), e(u)) * psi
    form += 2 * phi * p * div(u) * psi
    form += G_c * (- 1 / eps) * (1 - phi) * psi
    form += eps * InnerProduct(Grad(phi), Grad(psi))
    form += gamma * pos(phi - phi_last) * psi

    # ---------------------------- PHASE FIELD ----------------------------
    compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

    a = BilinearForm(X, symmetric=False)
    a += form.Compile(**compile_opts) * dx

    gf_phi_inv = GridFunction(W)
    gf_phi_inv.Set(1, definedon=mesh.Materials('crack'))
    gf_phi.Set(1 - gf_phi_inv)

    for i in range(pseudo_time_steps):
        gfu_last.vec.data = gfu.vec
        out = Newton(a, gfu, freedofs=X.FreeDofs(), inverse=newton_solver,
                     printing=newton_print, maxerr=newton_tol)
        print(i, out[1], Norm(gfu_last.vec - gfu.vec))

    return gfu


def sneddon_stationary(data, hmax, mesh_factor, eps, gamma, order=1,
                       kappa=1e-10, newton_tol=1e-8, newton_print=False,
                       newton_solver='umfpack', pseudo_time_steps=5,
                       compile_flag=False, wait_compile=False, **kwargs):
    '''
    We use a phase-field approach to compute an opening crack. We then
    construct a level set of the crack geometry by matching the crack
    volume computed from the phase field with the level set crack volume
    using a Newton scheme.

    Parameters
    ----------
    data : dict{'l0', 'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.
    hmax : float
        Maximal mesh size at fracture.
    mesh_factor : float
        Factor by which the outer domain is coarser by.
    eps : float
        Phase field parameter.
    gamma : float
        Constraint penalty parameter.
    order : float
        Polynomial order of phase field and solid finite element spaces.
    kappa : float
        Regularisation parameter.
    newton_tol : float
        Tolerance for phase field Newton scheme.
    newton_print : boolean
        Print output from phase filed Newton scheme.
    newton_solver : string
        Direct solver to use in Newton scheme.
    pseudo_time_steps : int
        Number of pseudo time step for which we solve the phase-field
        problem.
    compile_flag : boolean
        Strong compile weak form.
    wait_compile : boolean
        Wait for cpp compile to complete.

    Returns
    -------
    gfu : ngsolve.GridFunction
        Resulting finite element function containing the phase-field and
        displacement field.

    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'sneddon_stationary', category=SyntaxWarning)

    l0 = data['l0']

    # --------------------------------- MESH ----------------------------------
    geo = SplineGeometry()
    bc_dir = 'out'
    _l = max(2 * hmax, 0.01)
    geo.AddRectangle((0, 0), (4, 4), bcs=[bc_dir, bc_dir, bc_dir, bc_dir],
                     leftdomain=3, rightdomain=0)
    geo.AddRectangle((2 - l0 - _l, 2 - _l), (2 + l0 + _l, 2 + _l),
                     leftdomain=2, rightdomain=3)
    geo.AddRectangle((2 - l0, 2 - hmax), (2 + l0, 2 + hmax), leftdomain=1,
                     rightdomain=2)

    geo.SetMaterial(1, 'crack')
    geo.SetDomainMaxH(3, mesh_factor * hmax)
    geo.SetDomainMaxH(2, hmax)
    geo.SetDomainMaxH(1, hmax)

    mesh = Mesh(geo.GenerateMesh(grading=0.25, quad_dominated=False))

    # -------------------------- PHASE-FIELD PROBLEM --------------------------
    gfu = phase_field_stationary(data, mesh, bc_dir, eps, gamma, order,
                                 kappa, newton_tol, newton_print,
                                 newton_solver, pseudo_time_steps,
                                 compile_flag, wait_compile, **kwargs)

    return gfu


def sneddon_T_stationary(data, hmax, mesh_factor, eps, gamma, order=1,
                         kappa=1e-10, newton_tol=1e-8, newton_print=False,
                         newton_solver='umfpack', pseudo_time_steps=5,
                         compile_flag=False, wait_compile=False, **kwargs):
    '''
    We use a phase-field approach to compute an opening crack. We then
    construct a level set of the crack geometry by matching the crack
    volume computed from the phase field with the level set crack volume
    using a Newton scheme.

    Parameters
    ----------
    data : dict{'l0', 'G_c', 'E', 'nu_s', 'p'}
        Physical parameters determining Sneddon's test.
    hmax : float
        Maximal mesh size at fracture.
    mesh_factor : float
        Factor by which the outer domain is coarser by.
    eps : float
        Phase field parameter.
    gamma : float
        Constraint penalty parameter.
    order : float
        Polynomial order of phase field and solid finite element spaces.
    kappa : float
        Regularisation parameter.
    newton_tol : float
        Tolerance for phase field Newton scheme.
    newton_print : boolean
        Print output from phase filed Newton scheme.
    newton_solver : string
        Direct solver to use in Newton scheme.
    pseudo_time_steps : int
        Number of pseudo time step for which we solve the phase-field
        problem.
    compile_flag : boolean
        Strong compile weak form.
    wait_compile : boolean
        Wait for cpp compile to complete.

    Returns
    -------
    gfu : ngsolve.GridFunction
        Resulting finite element function containing the phase-field and
        displacement field.

    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'sneddon_T_stationary', category=SyntaxWarning)

    l0 = data['l0']
    bc_dir = 'outer'

    # --------------------------------- MESH ----------------------------------
    base = MoveTo(0, 0).Rectangle(4, 4).Face()
    crack = MoveTo(2 - l0, 2 - hmax).Line(2 * l0 - hmax, 0).Line(0, -l0 + hmax)
    crack = crack.Line(2 * hmax, 0).Line(0, 2 * l0).Line(-2 * hmax, 0)
    crack = crack.Line(0, -l0 + hmax).Line(-2 * l0 + hmax, 0).Close().Face()
    base -= crack

    base.faces.name = "material"
    base.faces.maxh = mesh_factor * hmax
    crack.faces.name = "crack"
    crack.faces.maxh = hmax

    geo = Glue([base, crack])
    geo.edges.name = bc_dir

    geo = OCCGeometry(geo, dim=2)
    mesh = Mesh(geo.GenerateMesh(grading=0.15, quad_dominated=False))

    # -------------------------- PHASE-FIELD PROBLEM --------------------------
    gfu = phase_field_stationary(data, mesh, bc_dir, eps, gamma, order,
                                 kappa, newton_tol, newton_print,
                                 newton_solver, pseudo_time_steps,
                                 compile_flag, wait_compile, **kwargs)

    return gfu


def tcv_from_phase_field(gfu):
    '''
    Compute the total crack volume from a phase-field finite element solution.
    '''

    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    order = gf_u.space.globalorder
    return Integrate(InnerProduct(gf_u, Grad(gf_phi)).Compile(), mesh,
                     order=2 * order - 1)


def cod_from_phase_field(gfu, lines, vertical=False):
    '''
    Compute the crack opening displacements by integrating along a line
    through the phase-field.

    Parameters
    ----------
    gfu : ngsolve.GridFunction
        Finite element solution of the displacement and phase-field.
    lines : list
        List of x-Coordinates at which we integrate over the y-domain.

    Returns
    -------
    crack_openings : list[tuple]
        List of tuples containing the x-coordinate and the corresponding
        crack aperture.
    '''
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    order = gf_u.space.globalorder

    _x = CF(x)
    if vertical:
        _x = CF(y)

    # Compute crack opening width based on phase-field
    lsetp1_line = GridFunction(H1(mesh, order=1))
    InterpolateToP1(_x - 2, lsetp1_line)
    ci_line = CutInfo(mesh, lsetp1_line)
    el_line = ci_line.GetElementsOfType(IF)
    ds_line = dCut(lsetp1_line, IF, order=2 * order, definedonelements=el_line)

    line_ind = InnerProduct(gf_u, Grad(gf_phi)).Compile()

    crack_openings = []
    for x0 in lines:
        InterpolateToP1(_x - x0, lsetp1_line)
        ci_line.Update(lsetp1_line)
        _cod = Integrate(line_ind * ds_line, mesh)

        crack_openings.append((x0, _cod))

    return crack_openings


def cod_from_phase_field_lset(gfu, lines):
    '''
    Compute the crack opening displacements by evaluating the
    displacement at the phase-field iso-line (1 + sqrt(5)) / 2 - 1).

    Parameters
    ----------
    gfu : ngsolve.GridFunction
        Finite element solution of the displacement and phase-field.
    lines : list
        List of x-Coordinates at which we integrate over the y-domain.

    Returns
    -------
    crack_openings : list[tuple]
        List of tuples containing the x-coordinate and the corresponding
        crack aperture.
    '''
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    V1 = H1(mesh, order=1)

    lsetp1, lsetp1_line = GridFunction(V1), GridFunction(V1)
    InterpolateToP1(gf_phi - ((1 + sqrt(5)) / 2 - 1), lsetp1)
    InterpolateToP1(x - 2, lsetp1_line)

    lsetsp1 = (lsetp1, lsetp1_line)
    mlci_cod = MultiLevelsetCutInfo(mesh, lsetsp1)
    dtt_pnts = DomainTypeArray((IF, IF))
    els_pnts = mlci_cod.GetElementsWithContribution(dtt_pnts)
    dx_pnt = dCut(lsetsp1, dtt_pnts, order=0, definedonelements=els_pnts)

    n_lset = Grad(lsetp1) / Norm(Grad(lsetp1))
    u_n = InnerProduct(gf_u, n_lset)

    crack_openings = []
    for x0 in lines:
        InterpolateToP1(x - x0, lsetp1_line)
        mlci_cod.Update(lsetsp1)
        _cod = Integrate(u_n * dx_pnt, mesh)
        crack_openings.append((x0, _cod))

    return crack_openings


def _line_cf(p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    a = - (y0 - y1)
    b = (x0 - x1)
    c = - y0 * (x0 - x1) + x0 * (y0 - y1)
    return a, b, c


def _makeline(p0, p1):
    a, b, c = _line_cf(p0, p1)
    return a * x + b * y + c


def _min_cf(a, b):
    return IfPos(b - a, b, a)


def _make_lsets_to_one(lsets):
    n = len(lsets)
    if n == 1:
        return lsets[0]
    else:
        lsets_new = []
        for _i in range(0, len(lsets) - 1, 2):
            lsets_new.append(_min_cf(lsets[_i], lsets[_i + 1]))
        if _i < n - 2:
            lsets_new.append(lsets[-1])
        return _make_lsets_to_one(lsets_new)


def make_levelset_from_cod_convex(pnt_and_cod, mesh):
    '''
    Create a piecewise linear level set function based on a list of
    x-coordinates and crack apertures. Assumes the set of points build a
    convex set.

    Parameters
    ----------
    pnt_and_cod : list[tuple]
        List of x-coordinates and crack apertures.
    mesh : ngsolve.Mesh
        The mesh to construct the level set on.

    Returns
    -------
    ngsolve.GridFunction
        The piecewise linear level set function.
    '''
    lset_sections = []

    for i in range(len(pnt_and_cod) - 1):
        _x0, _h0 = pnt_and_cod[i]
        _x1, _h1 = pnt_and_cod[i + 1]
        _line1 = -_makeline((_x0, 2 + _h0 / 2), (_x1, 2 + _h1 / 2))
        _line2 = _makeline((_x0, 2 - _h0 / 2), (_x1, 2 - _h1 / 2))
        lset_sections.append(_min_cf(_line1, _line2))

    lsetp1 = GridFunction(H1(mesh, order=1))
    InterpolateToP1(_make_lsets_to_one(lset_sections), lsetp1)

    return lsetp1


def make_multiple_levelset_from_cod(pnt_and_cod, mesh):
    '''
    Make a multiple level set description of a set of x-coordinates and
    crack apertures.

    Parameters
    ----------
    pnt_and_cod : list[tuple]
        List of x-coordinates and crack apertures.
    mesh : ngsolve.Mesh
        The mesh to construct the level set on.

    Returns
    -------
    xfem.mlset.DomainTypeArray
        Container of the level sets and domain description.
    '''
    lsets = []
    mlast = 0
    for i in range(len(pnt_and_cod) - 1):
        _x0, _h0 = pnt_and_cod[i]
        _x1, _h1 = pnt_and_cod[i + 1]

        a, b, c = _line_cf((_x0, 2 + _h0 / 2), (_x1, 2 + _h1 / 2))
        m = -a / b
        if m > mlast:
            lsets.append([])

        _line1 = -_makeline((_x0, 2 + _h0 / 2), (_x1, 2 + _h1 / 2))
        _line2 = _makeline((_x0, 2 - _h0 / 2), (_x1, 2 - _h1 / 2))
        lsets[-1].append((_min_cf(_line1, _line2), _x1))
        mlast = m

    P1 = H1(mesh, order=1)
    _outer, _divides = [], []

    for llist in lsets:
        ll = [_l[0] for _l in llist]
        section = _make_lsets_to_one(ll)
        _outer.append(GridFunction(P1))
        InterpolateToP1(section, _outer[-1])
        _divides.append(GridFunction(P1))
        InterpolateToP1(x - llist[-1][1], _divides[-1])

    _divides.pop()
    all_lsets = tuple(_outer + _divides)
    mlset_doms = []
    for i in range(len(_outer)):
        dom1 = [ANY for j in range(len(_outer))]
        dom1[i] = NEG

        dom2 = [NEG for j in range(len(_divides))]
        for j in range(i):
            dom2[j] = POS

        mlset_doms.append(tuple(dom1 + dom2))

    return DomainTypeArray(mlset_doms, all_lsets, True)


def make_mesh_from_points(_pts, hmax, curvaturesafety=0.7):
    ''''
    Create a rational spline geometry and mesh given a set of points on
    the boundary.

    Parameters
    ----------
    _pts : list(tuple)
        Coordinates of the points on the boundary.
    hmax : float
        Mesh size
    curvaturesafety : float
        Ensure that curvature of elements id not too large.

    Returns
    -------
    ngsolve.Mesh
    '''
    control_polygon = []
    for p0, p1, p2 in zip([_pts[-1]] + _pts[:-1], _pts, _pts[1:] + [_pts[0]]):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        if abs((x2 - x0)) < 1e-10:
            a = 1
            b = 0
            c = x1
        else:
            a = (y0 - y2) / (x2 - x0)
            b = 1
            c = a * x1 + y1

        control_polygon.append((a, b, c))

    control_points = []
    control_polygon = control_polygon + [control_polygon[0]]

    for (a1, a2) in pairwise(control_polygon):
        a = np.array([a1[0:2], a2[0:2]])
        b = np.array([a1[2], a2[2]])
        control_points.append(tuple(np.linalg.solve(a, b)))

    geo = SplineGeometry()
    pts = [geo.AppendPoint(*p) for p in _pts]
    c_pts = [geo.AppendPoint(*p) for p in control_points]

    for (p0, p1), p2 in zip(pairwise(pts), c_pts[:-1]):
        geo.Append(["spline3", p0, p2, p1], bc='crack',
                   leftdomain=0, rightdomain=1)
    geo.Append(["spline3", pts[-1], c_pts[-1], pts[0]], bc='crack',
               leftdomain=0, rightdomain=1)

    return Mesh(geo.GenerateMesh(maxh=hmax, curvaturesafety=curvaturesafety))


def make_fsi_mesh_from_points(out_pts, _pts, hmax, solid_fac, curvaturesafety):
    ''''
    Create a mesh given a set of points of a fluid domain contained
    inside a solid domain.

    Parameters
    ----------
    out_pts : list(tuple)
        Coordinates of the outer solid boundary.
    _pts : list(tuple)
        Coordinates of the points on the boundary of the fluid domain.
    hmax : float
        Mesh size
    curvaturesafety : float
        Ensure that curvature of elements id not too large.

    Returns
    -------
    ngsolve.Mesh
    '''
    control_polygon = []
    for p0, p1, p2 in zip([_pts[-1]] + _pts[:-1], _pts, _pts[1:] + [_pts[0]]):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        if abs((x2 - x0)) < 1e-10:
            a = 1
            b = 0
            c = x1
        else:
            a = (y0 - y2) / (x2 - x0)
            b = 1
            c = a * x1 + y1

        control_polygon.append((a, b, c))

    control_points = []
    control_polygon = control_polygon + [control_polygon[0]]

    for (a1, a2) in pairwise(control_polygon):
        a = np.array([a1[0:2], a2[0:2]])
        b = np.array([a1[2], a2[2]])
        control_points.append(tuple(np.linalg.solve(a, b)))

    geo = SplineGeometry()
    geo.AddRectangle(out_pts[0], out_pts[1], bcs=['out', 'out', 'out', 'out'],
                     leftdomain=1, rightdomain=0)

    pts = [geo.AppendPoint(*p) for p in _pts]
    c_pts = [geo.AppendPoint(*p) for p in control_points]

    for (p0, p1), p2 in zip(pairwise(pts), c_pts[:-1]):
        geo.Append(["spline3", p0, p2, p1], bc='interface',
                   leftdomain=1, rightdomain=2, maxh=hmax)
    geo.Append(["spline3", pts[-1], c_pts[-1], pts[0]], bc='interface',
               leftdomain=1, rightdomain=2, maxh=hmax)

    geo.SetMaterial(1, 'solid')
    geo.SetMaterial(2, 'fluid')
    geo.SetDomainMaxH(1, hmax * solid_fac)
    geo.SetDomainMaxH(2, hmax)

    return Mesh(geo.GenerateMesh(grading=0.25, curvaturesafety=curvaturesafety))


def _compute_deformation(gf_u, lsetp1, dirichlet, order_def, inverse,
                         frac_def=0.5, frac_base=1, sigma_def=100, **kwargs):
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'compute_deformation', category=SyntaxWarning)

    n_lset = 1.0 / Norm(grad(lsetp1)) * grad(lsetp1)
    h = specialcf.mesh_size
    _mesh = gf_u.space.mesh

    ci = CutInfo(_mesh, lsetp1)
    els_if = ci.GetElementsOfType(IF)
    ds_if = dCut(lsetp1, IF, definedonelements=els_if)

    V2 = VectorH1(_mesh, order=order_def, dirichlet=dirichlet)
    deformation = GridFunction(V2)

    u, v = V2.TnT()
    poisson_vol = InnerProduct(Grad(u), Grad(v)).Compile()
    poisson_bnd = - InnerProduct(Grad(u) * n_lset, v)
    poisson_bnd += - InnerProduct(u, Grad(v) * n_lset)
    poisson_bnd += sigma_def * order_def**2 / h * InnerProduct(u, v)
    poisson_bnd = poisson_bnd.Compile()

    u_D = frac_def * InnerProduct(gf_u, n_lset) * CF((0, IfPos(y - 2, 1, -1)))
    u_D += frac_base * CF((0, -(y - 2)))
    poisson_rhs = - InnerProduct(u_D, Grad(v) * n_lset)
    poisson_rhs += sigma_def * order_def**2 / h * InnerProduct(u_D, v)
    poisson_rhs = poisson_rhs.Compile()

    _a = BilinearForm(V2)
    _a += poisson_vol * dx
    _a += poisson_bnd * ds_if

    _f = LinearForm(V2)
    _f += poisson_rhs * ds_if

    _f.Assemble()
    _a.Assemble()
    _a_inv = _a.mat.Inverse(V2.FreeDofs(), inverse=inverse)

    deformation.vec.data = PreconditionedRichardson(_a, _f.vec, pre=_a_inv,
                                                    freedofs=V2.FreeDofs(),
                                                    printing=False)

    return deformation


def make_levelset_from_transport(gfu, lsetp1, dt, order_def, bc_dir, inverse,
                                 compile_flag=False, wait_compile=False,
                                 **kwargs):
    '''
    Compute an Eulerian level set description of a phase-field crack
    by solving a level set transport problem.

    Parameters
    ----------
    gfu : ngsolve.GridFunction
        The finite element function containing the displacement and
        phase-field.
    lsetp1 : ngsolve.GridFunction
        The level set to be transported.
    dt : float
        The transport equation time step.
    order_def : int
        Order of the finite element deformation by which the level set
        is transported.
    bc_dir : string
        The dirichlet boundary condition for the deformation on the
        underlying mesh.
    inverse : string
        Direct solver to use
    compile_flag : bool
        Compile cpp code.
    wait_compile : bool
        Wait for cpp compilation to complete.

    Returns
    -------
    lset_trans : ngsolve.GridFunction

    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'make_levelset_from_transport', category=SyntaxWarning)

    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    V1 = lsetp1.space

    deformation = _compute_deformation(gf_u=gf_u, lsetp1=lsetp1,
                                       order_def=order_def, dirichlet=bc_dir,
                                       inverse=inverse,
                                       frac_def=1, frac_base=1)
    h = specialcf.mesh_size
    gam_t = 0.5 * h / 1
    compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

    V2 = H1(mesh, order=order_def)

    wind1, wind2 = GridFunction(V2), GridFunction(V2)
    wind = CF((wind1, wind2))

    u1, v1 = V1.TnT()
    form_a_lset = wind * Grad(u1) * (v1 + gam_t * wind * Grad(v1))
    form_a_lset = form_a_lset.Compile(**compile_opts)
    form_ms_lset = (u1 + dt * wind * Grad(u1)) * (v1 + gam_t * wind * Grad(v1))
    form_ms_lset = form_ms_lset.Compile(**compile_opts)

    a = BilinearForm(V1, nonassemble=True)
    a += form_a_lset * dx

    mstar1 = BilinearForm(V1)
    mstar1 += form_ms_lset * dx

    u2, v2 = V2.TnT()
    form_a_wind = wind * Grad(u2) * (v2 + gam_t * wind * Grad(v2))
    form_a_wind = form_a_wind.Compile(**compile_opts)
    form_ms_wind = (u2 + dt * wind * Grad(u2)) * (v2 + gam_t * wind * Grad(v2))
    form_ms_wind = form_ms_wind.Compile(**compile_opts)

    a2 = BilinearForm(V2, nonassemble=True)
    a2 += form_a_wind * dx

    mstar2 = BilinearForm(V2)
    mstar2 += form_ms_wind * dx

    lset_trans = GridFunction(V1)
    res1, res2 = lset_trans.vec.CreateVector(), wind1.vec.CreateVector()
    wind1.Set(deformation[0])
    wind2.Set(deformation[1])

    lset_trans.vec.data = lsetp1.vec

    for i in range(int(1 / dt)):
        mstar1.Assemble()
        inv_mstar1 = mstar1.mat.Inverse(inverse=inverse)

        a.Apply(lset_trans.vec, res1)
        lset_trans.vec.data -= dt * inv_mstar1 * res1

        mstar2.Assemble()
        inv_mstar2 = mstar2.mat.Inverse(inverse=inverse)

        a2.Apply(wind1.vec, res2)
        wind1.vec.data -= dt * inv_mstar2 * res2
        a2.Apply(wind2.vec, res2)
        wind2.vec.data -= dt * inv_mstar2 * res2

    return lset_trans


def tcv_from_eulerian_levelset(lsetp1):
    '''
    Compute the area of the negative region of a given level set.

    Parameters
    ----------
    lsetp1 : ngsolve.GridFunktion
        The piecewise linear level set function

    Returns
    -------
    area : float
    '''

    mesh = lsetp1.space.mesh
    ci = CutInfo(mesh, lsetp1)
    els_hasneg = ci.GetElementsOfType(HASNEG)
    dx_neg = dCut(lsetp1, NEG, order=1, definedonelements=els_hasneg)

    return Integrate(CF(1).Compile() * dx_neg, mesh)


def cod_from_eulerian_levelset(lsetp1, lines):
    mesh = lsetp1.space.mesh

    lsetp1_line = GridFunction(H1(mesh, order=1))
    InterpolateToP1(x - 2, lsetp1_line)
    lsetsp1 = (lsetp1, lsetp1_line)

    mlci_cod = MultiLevelsetCutInfo(mesh, lsetsp1)
    dtt_line = DomainTypeArray((NEG, IF))
    els_cod = mlci_cod.GetElementsWithContribution(dtt_line)
    ds_cod = dCut(lsetsp1, dtt_line, order=1, definedonelements=els_cod)

    one_cf = CF(1).Compile()

    crack_openings = []
    for x0 in lines:
        InterpolateToP1(x - x0, lsetp1_line)
        mlci_cod.Update(lsetsp1)
        _cod = Integrate(one_cf * ds_cod, mesh)

        crack_openings.append((x0, _cod))

    return crack_openings


def tcv_from_eulerian_multiple_levelset(mlset_dom):
    '''
    Compute the area of a multiple level set domain.

    Parameters
    ----------
    mlset_dom : xfem.mlset.DomainTypeArray
        The multiple level set description of the domain.

    Returns
    -------
    area : float
    '''
    mesh = mlset_dom.lsets[0].space.mesh
    mlci = MultiLevelsetCutInfo(mesh, mlset_dom.lsets)
    els_hasneg = mlci.GetElementsWithContribution(mlset_dom)
    dx_neg = dCut(mlset_dom.lsets, mlset_dom, order=1,
                  definedonelements=els_hasneg)

    return Integrate(CF(1).Compile() * dx_neg, mesh)


def cod_from_eulerian_multiple_levelset(mlset_dom, lines):
    '''
    Compute the crack opening displacements based on a multiple level
    set description of the domain.

    Parameters
    ----------
    mlset_dom : xfem.mlset.DomainTypeArray
        The multiple level set description of the domain.
    lines : list
        List of x-Coordinates at which we integrate over the y-domain.

    Returns
    -------
    crack_openings : list[tuple]
        List of tuples containing the x-coordinate and the corresponding
        crack aperture.
    '''
    mesh = mlset_dom.lsets[0].space.mesh

    lsetp1_line = GridFunction(H1(mesh, order=1))
    InterpolateToP1(x - 2, lsetp1_line)
    lsets_all = tuple(list(mlset_dom.lsets) + [lsetp1_line])

    dta_line = DomainTypeArray(IF)
    dta = TensorIntersection(mlset_dom, dta_line)

    mlci_cod = MultiLevelsetCutInfo(mesh, lsets_all)

    els_cod = mlci_cod.GetElementsWithContribution(dta)
    ds_cod = dCut(lsetsp1, dta, order=1, definedonelements=els_cod)

    one_cf = CF(1).Compile()

    crack_openings = []
    for x0 in lines:
        InterpolateToP1(x - x0, lsetp1_line)
        mlci_cod.Update(lsets_all)
        _cod = Integrate(one_cf * ds_cod, mesh)

        crack_openings.append((x0, _cod))

    return crack_openings


def non_convex_points(points):
    '''
    Given a set of points, return a(n incomplete) list of indices of
    points that destroy the convexity of the domain connecting the
    points.
    '''
    remove = []
    for i in range(len(points) - 2):
        p0, p1, p2 = points[i], points[i + 1], points[i + 2]
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        m1 = (y1 - y0) / (x1 - x0)
        m2 = (y2 - y1) / (x2 - x1)
        if m1 < m2:
            remove.append(i + 1)
    return remove
