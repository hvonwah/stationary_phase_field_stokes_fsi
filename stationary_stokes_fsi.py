from ngsolve import *


def stationary_stokes_fsi(mesh, order, data, rhs_data_fluid, alpha_u=1e-14,
                          harmonic_extension=True, compile_flag=False,
                          newton_print=True, newton_damp=1, newton_tol=1e-15,
                          inverse='pardiso', condense=True):
    '''
    Compute a stationary Stokes fluid-structure interaction problem.

    Parameters
    ----------
    mesh : ngsolve.Mesh
        Mesh with a 'fluid' and 'solid' regions, the boundary condition
        marker 'interface' between them and boundary marker 'out' for
        the outer homogeneous Dirichlet condition of the solid.
    order : int
        Order for the velocity and displacement finite element spaces.
    data : dict
        Dictionary containing the fluid and solid parameters.
    rhs_data_fluid : ngsolve.CoefficientFunction
        Right-hand side data for the fluid problem.
    alpha_u : float
        Harmonic extension parameter
    harmonic_extension : bool
        Use harmonic extension for displacements into fluid.
    compile_flag : bool
        Compile cpp code.
    newton_print : bool
        Print output from Newton solver.
    newton_damp : float
        Factor for Newton damping.
    newton_tol : float
        Residual tolerance for Newton solver.
    inverse : string
        Direct solver to use in Newton scheme.
    condense : bool
        Apply static condensation of internal bubbles for higher-order
        elements.

    Returns
    -------
    gfu, drag, lift : tuple(ngsolve.GridFunction, float, float)
        The FSI solution, drag and lift acting on the interface.
    '''
    mus = data['mus']
    ls = data['lams']
    rhof = data['rhof']
    nuf = data['nuf']

    # ------------------------- FINITE ELEMENT SPACE --------------------------
    V = VectorH1(mesh, order=order, dirichlet='out')
    Q = H1(mesh, order=order - 1, definedon='fluid')
    D = VectorH1(mesh, order=order, dirichlet='out')
    X = FESpace([V, Q, D])
    Y = FESpace([V, Q])
    (u, p, d), (v, q, w) = X.TnT()

    gfu = GridFunction(X)

    print(f'Number of FreeDofs of product space = {sum(X.FreeDofs(condense))}')

    # --------------------------- (BI)LINEAR FORMS ----------------------------
    Id2 = Id(2)
    F = Grad(d) + Id2
    C = F.trans * F
    E = 0.5 * (Grad(d) + Grad(d).trans)
    J = Det(F)
    Finv = Inv(F)
    FinvT = Finv.trans

    stress_sol = 2 * mus * E + ls * Trace(E) * Id2
    stress_fl = rhof * nuf * (grad(u) * Finv + FinvT * grad(u).trans)

    diff_fl = rhof * nuf * InnerProduct(J * stress_fl * FinvT, grad(v))
    pres_fl = -J * (Trace(grad(v) * Finv) * p + Trace(grad(u) * Finv) * q)
    pres_fl += - J * 1e-9 * p * q

    rhs_fl = - InnerProduct(rhof * J * rhs_data_fluid, v)

    mass_sol = - InnerProduct(u, w)
    el_sol = InnerProduct(J * stress_sol * FinvT, grad(v))

    if harmonic_extension:
        extension = alpha_u * InnerProduct(Grad(d), Grad(d))
    else:
        gfdist = GridFunction(H1(mesh, order=1, dirichlet='out'))
        gfdist.Set(1, definedon=mesh.Boundaries('interface'))

        def NeoHookExt(C, mu=1, lam=1):
            ext = 0.5 * mu * Trace(C - Id2)
            ext += 0.5 * mu * (T2 * mu / lam * Det(C)**(-lam / 2 / mu) - 1)
            return ext

        extension = 1 / (1 - gfdist + 1e-2) * 1e-8 * NeoHookExt(C)

    stokes = nuf * rhof * InnerProduct(grad(u), grad(v))
    stokes += - div(u) * q - div(v) * p - 1e-9 * p * q

    # ------------------------------ INTEGRATORS ------------------------------
    comp_opt = {'realcompile': compile_flag, 'wait': True}
    dFL, dSL = dx('fluid'), dx('solid')

    a = BilinearForm(X, symmetric=False, condense=condense)
    a += (diff_fl + pres_fl + rhs_fl).Compile(**comp_opt) * dFL
    a += (mass_sol + el_sol).Compile(**comp_opt) * dSL
    a += Variation(extension.Compile(**comp_opt) * dFL)

    a_stokes = BilinearForm(Y, symmetric=True, check_unused=False)
    a_stokes += stokes * dFL

    f_stokes = LinearForm(Y)
    f_stokes += InnerProduct(rhs_data_fluid, v) * dFL

    # ------------------------ FUNCTIONAL COMPUTATION -------------------------
    gfu_drag, gfu_lift = GridFunction(X), GridFunction(X)
    gfu_drag.components[0].Set(CF((1, 0)),
                               definedon=mesh.Boundaries('interface'))
    gfu_lift.components[0].Set(CF((0, 1)),
                               definedon=mesh.Boundaries('interface'))
    res = gfu.vec.CreateVector()

    # ----------------------------- SOLVE PROBLEM -----------------------------
    bts = Y.FreeDofs() & ~Y.GetDofs(mesh.Materials('solid'))
    bts &= ~Y.GetDofs(mesh.Boundaries('out|interface'))
    bts[Y.Range(1)] = True

    rstokes = GridFunction(Y)

    a_stokes.Assemble()
    f_stokes.Assemble()
    invstoke = a_stokes.mat.Inverse(bts, inverse='sparsecholesky')

    rstokes.vec.data = invstoke * f_stokes.vec

    gfu.components[0].vec.data = rstokes.components[0].vec
    gfu.components[1].vec.data = rstokes.components[1].vec

    solvers.Newton(a, gfu, maxit=10, inverse=inverse, maxerr=newton_tol,
                   dampfactor=newton_damp, printing=newton_print)

    a.Apply(gfu.vec, res)
    drag = - InnerProduct(res, gfu_drag.vec)
    lift = - InnerProduct(res, gfu_lift.vec)

    return gfu, drag, lift
