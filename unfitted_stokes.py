from ngsolve import VectorH1, NumberSpace, FESpace, GridFunction, \
    specialcf, Norm, Grad, div, InnerProduct, LinearForm, dx, sqrt, \
    Integrate, BilinearForm, H1
from ngsolve.solvers import PreconditionedRichardson
from xfem import Restrict, CutInfo, GetFacetsWithNeighborTypes, \
    dCut, NEG, IF, HASNEG, HAS, dFacetPatch, MultiLevelsetCutInfo
from xfem.mlset import *
import warnings


def refine_levelsetdomain(mesh, lsetp1, DOM=NEG, nref=1):
    '''
    Refine Mesh elements which contain part of the DOM domain
    '''
    for i in range(nref):
        ci = CutInfo(mesh, lsetp1)
        mesh.SetRefinementFlags(ci.GetElementsOfType(HAS(DOM)))
        mesh.Refine()
    return None


def L2Norm(lset, u):
    '''
    Compute the L2-norm over the NEG part of the level set domain.
    '''
    return sqrt(Integrate(InnerProduct(u, u).Compile() * dCut(lset, NEG),
                          lset.space.mesh))


def L2Norm_mlset(dta, u):
    '''
    Compute the L2-norm over the NEG part of the level set domain.
    '''
    return sqrt(Integrate(InnerProduct(u, u).Compile() * dCut(dta.lsets, dta),
                          dta.lsets[0].space.mesh))


def solve_unfitted_stokes(lsetp1, nu, gamma_n=100, gamma_gp=0.1,
                          f_vol=None, g_bnd=None, compile_flag=False,
                          wait_compile=False, inverse='umfpack', **kwargs):
    '''
    Parameters
    ----------
    lsetp1 : GridFunction(H1)
        The piecewise linear level set function describing the domain.
    nu : float
        Viscosity parameter in the Stokes problem.
    gamma_n : float, default
        Nitsche penalty parameter to enforce the boundary condition.
    gamma_gp : float
        Ghost penalty stabilisation parameter.
    f_vol : ngsolve.CoefficientFunction(dim=2)
        Volumetric forcing term. Either this or a non-homogeneous
        Dirichlet boundary condition must be provided.
    g_bnd : ngsolve.CoefficientFunction(dim=2)
        Inhomogeneous boundary condition for the velocity.
    compile_flag : bool
        Hard (c++) compile the integrator forms.
    wait_compile : bool
        Wait for hard compile to complete before continue.
    inverse : string
        Sparse direct solver for the resulting linear system.

    Returns
    -------
    gfu : ngsolve.GridFunction([VectorH1, H1, NumberSpace])
        Solution to the Stokes problem.
    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'solve_unfitted_stokes', category=SyntaxWarning)

    if f_vol is None and g_bnd is None:
        raise Exception('No right-hand side data for Stokes problem provided!')

    mesh = lsetp1.space.mesh
    ci = CutInfo(mesh, lsetp1)
    els_hasneg = ci.GetElementsOfType(HASNEG)
    els_if = ci.GetElementsOfType(IF)
    facets_gp = GetFacetsWithNeighborTypes(mesh, a=els_hasneg, b=els_if,
                                           use_and=True)

    Vbase = VectorH1(mesh, order=2)
    V = Restrict(Vbase, els_hasneg)
    Qbase = H1(mesh, order=1)
    Q = Restrict(Qbase, els_hasneg)
    N = NumberSpace(mesh)
    X = FESpace([V, Q, N], dgjumps=True)

    active_dofs = X.FreeDofs()
    print(f'Number of FreeDofs of product space = {sum(active_dofs)}')

    gfu = GridFunction(X)

    (u, p, lam), (v, q, mu) = X.TnT()
    h = specialcf.mesh_size
    n_lset = Grad(lsetp1) / Norm(Grad(lsetp1))

    compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

    stokes_vol = nu * InnerProduct(Grad(u), Grad(v))
    stokes_vol += - p * div(v) - q * div(u) + p * mu + q * lam

    stokes_bnd = - nu * InnerProduct(Grad(u) * n_lset, v)
    stokes_bnd += - nu * InnerProduct(Grad(v) * n_lset, u)
    stokes_bnd += nu * gamma_n / h * InnerProduct(u, v)
    stokes_bnd += p * InnerProduct(v, n_lset)
    stokes_bnd += q * InnerProduct(u, n_lset)

    if f_vol is not None:
        rhs_vol = InnerProduct(f_vol, v)
    if g_bnd is not None:
        rhs_bnd = - nu * InnerProduct(Grad(v) * n_lset, g_bnd)
        rhs_bnd += nu * gamma_n / h * InnerProduct(g_bnd, v)
        rhs_bnd += q * InnerProduct(g_bnd, n_lset)

    ghost_penalty = nu / h**2 * InnerProduct(u - u.Other(), v - v.Other())
    ghost_penalty += -1 / nu * (p - p.Other()) * (q - q.Other())

    dx_neg = dCut(lsetp1, NEG, definedonelements=els_hasneg)
    ds_if = dCut(lsetp1, IF, definedonelements=els_if)
    dw = dFacetPatch(definedonelements=facets_gp)

    a = BilinearForm(X)
    a += stokes_vol.Compile(**compile_opts) * dx_neg
    a += stokes_bnd.Compile(**compile_opts) * ds_if
    a += ghost_penalty.Compile(**compile_opts) * dw
    a.Assemble()
    inv_a = a.mat.Inverse(active_dofs, inverse=inverse)

    f = LinearForm(X)
    if f_vol is not None:
        f += rhs_vol.Compile(**compile_opts) * dx_neg
    if g_bnd is not None:
        f += rhs_bnd.Compile(**compile_opts) * ds_if
    f.Assemble()

    gfu.vec.data = PreconditionedRichardson(a, f.vec, pre=inv_a,
                                            printing=False)

    del a, f, inv_a

    return gfu


def solve_unfitted_stokes_mlset(mlset_dom, nu, gamma_n=100, gamma_gp=0.1,
                                f_vol=None, g_bnd=None, compile_flag=False,
                                wait_compile=False, inverse='umfpack',
                                **kwargs):
    '''
    Parameters
    ----------
    mlset_dom : DomainTypeArray
        The the domain description together with the relevant levelsets.
    nu : float
        Viscosity parameter in the Stokes problem.
    gamma_n : float
        Nitsche penalty parameter to enforce the boundary condition.
    gamma_gp : float
        Ghost penalty stabilisation parameter.
    f_vol : ngsolve.CoefficientFunction(dim=2)
        Volumetric forcing term. Either this or a non-homogeneous
        Dirichlet boundary condition must be provided.
    g_bnd : ngsolve.CoefficientFunction(dim=2)
        Inhomogeneous boundary condition for the velocity.
    compile_flag : bool
        Hard (c++) compile the integrator forms.
    wait_compile : bool
        Wait for hard compile to complete before continue.
    inverse : string
        Sparse direct solver for the resulting linear system.

    Returns
    -------
    gfu : ngsolve.GridFunction([VectorH1, H1, NumberSpace])
        Solution to the Stokes problem.
    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'solve_unfitted_stokes', category=SyntaxWarning)

    if f_vol is None and g_bnd is None:
        raise Exception('No right-hand side data for Stokes problem provided!')

    mesh = mlset_dom.lsets[0].space.mesh
    lsets = mlset_dom.lsets
    boundary = mlset_dom.Boundary()

    mlci = MultiLevelsetCutInfo(mesh, lsets)

    els_hasneg = mlci.GetElementsWithContribution(mlset_dom)
    els_if = mlci.GetElementsWithContribution(boundary)
    facets_gp = GetFacetsWithNeighborTypes(mesh, a=els_hasneg, b=els_if,
                                           use_and=True)
    els_if_single = {}
    for i, dtt in enumerate(boundary):
        els_if_single[dtt] = mlci.GetElementsWithContribution(dtt)

    Vbase = VectorH1(mesh, order=2)
    V = Restrict(Vbase, els_hasneg)
    Qbase = H1(mesh, order=1)
    Q = Restrict(Qbase, els_hasneg)
    N = NumberSpace(mesh)
    X = FESpace([V, Q, N], dgjumps=True)

    active_dofs = X.FreeDofs()
    print(f'Number of FreeDofs of product space = {sum(active_dofs)}')

    gfu = GridFunction(X)

    (u, p, lam), (v, q, mu) = X.TnT()
    h = specialcf.mesh_size

    normals = mlset_dom.GetOuterNormals(lsets)

    compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

    stokes_vol = nu * InnerProduct(Grad(u), Grad(v))
    stokes_vol += - p * div(v) - q * div(u) + p * mu + q * lam

    ghost_penalty = nu / h**2 * InnerProduct(u - u.Other(), v - v.Other())
    ghost_penalty += -1 / nu * (p - p.Other()) * (q - q.Other())

    dx_neg = dCut(lsets, mlset_dom, definedonelements=els_hasneg)
    ds_if = {dtt: dCut(lsets, dtt, definedonelements=els_if_single[dtt])
             for dtt in boundary}
    dw = dFacetPatch(definedonelements=facets_gp)

    a = BilinearForm(X)
    a += stokes_vol.Compile(**compile_opts) * dx_neg
    for bnd, n in normals.items():
        stokes_bnd = - nu * InnerProduct(Grad(u) * n, v)
        stokes_bnd += - nu * InnerProduct(Grad(v) * n, u)
        stokes_bnd += nu * gamma_n / h * InnerProduct(u, v)
        stokes_bnd += p * InnerProduct(v, n)
        stokes_bnd += q * InnerProduct(u, n)
        a += stokes_bnd.Compile(**compile_opts) * ds_if[bnd]
    a += ghost_penalty.Compile(**compile_opts) * dw

    a.Assemble()
    inv_a = a.mat.Inverse(active_dofs, inverse=inverse)

    f = LinearForm(X)
    if f_vol is not None:
        rhs_vol = InnerProduct(f_vol, v)
        f += rhs_vol.Compile(**compile_opts) * dx_neg
    if g_bnd is not None:
        for bnd, n in normals.items():
            rhs_bnd = - nu * InnerProduct(Grad(v) * n, g_bnd)
            rhs_bnd += nu * gamma_n / h * InnerProduct(g, v)
            rhs_bnd += q * InnerProduct(g, n_lset)
            f += rhs_bnd.Compile(**compile_opts) * ds_if[bnd]
    f.Assemble()

    gfu.vec.data = PreconditionedRichardson(a, f.vec, pre=inv_a,
                                            printing=False)

    del a, f, inv_a

    return gfu
