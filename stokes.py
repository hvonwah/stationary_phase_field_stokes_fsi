from ngsolve import VectorH1, H1, NumberSpace, FESpace, GridFunction, CF, \
    InnerProduct, Grad, div, BilinearForm, LinearForm, dx, Integrate, sqrt, \
    specialcf
import warnings


def L2Norm(mesh, cf):
    '''
    Compute the L2 norm of a function on a mesh.

    Parameters
    ----------
    mesh : ngsolve.Mesh
        The mesh under consideration.
    cf : ngsolve.CoefficientFunction
        The function to take the norm of.

    Returns
    -------
    float : The L2 norm of the function.
    '''
    return sqrt(Integrate(InnerProduct(cf, cf) * dx(bonus_intorder=5), mesh))


def solve_stokes(mesh, order, dirichlet, lagr=False, f_vol=CF((0, 0)), nu=1,
                 g_bnd=CF((0, 0)), g_bnd_dom=None, compile_flag=False,
                 wait_compile=False, inverse='umfpack', condense=True,
                 **kwargs):
    '''
    Solve Stokes problem using Taylor-Hood elements on a given mesh.

    Parameters
    ----------
    mesh : ngsolve.GridFunction
        The mesh under consideration.
    order : int
        The polynomia order of the velocity space=.
    dirichlet : string
        List of Dirichlet boundaries.
    lagr : bool
        Use a Lagrange multiplier to enforce mean zero of the pressure.
    f_vol : ngsolve.CoefficientFunction
        Volumetric right-hand side data.
    nu : float
        Viscosity.
    g_bnd : ngsolve.CoefficientFunction
        Right-hand side boundary data.
    g_bnd_dom : string or None
        Boundary domain on which to apply the boundary data.
    compile_flag : bool
        Compile cpp code
    wait_compile : bool
        Wait for cpp code to finish compiling.
    inverse : string
        Direct solver to use.
    condense : bool
        Use static condensation of internal bubbles for higher-order
        finite elements.

    Returns
    -------
    tuple(ngsolve.GridFunction, ngsolve.GridFunction)
        Velocity and pressure solution.
    '''
    for key, value in kwargs.items():
        warnings.warn(f'Unknown keyword argument {key}={value} in call of '
                      + 'sneddon_stationary', category=SyntaxWarning)

    V = VectorH1(mesh, order=order, dirichlet=dirichlet)
    Q = H1(mesh, order=order - 1)
    if lagr:
        N = NumberSpace(mesh)
        X = FESpace([V, Q, N])
        (u, p, lam), (v, q, mu) = X.TnT()
    else:
        X = FESpace([V, Q], dgjumps=True)
        (u, p), (v, q) = X.TnT()
    print(f'Number of FreeDofs of product space = {sum(X.FreeDofs(condense))}')

    gfu = GridFunction(X)
    compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

    stokes = nu * InnerProduct(Grad(u), Grad(v)) - p * div(v) - q * div(u)
    if lagr:
        stokes += lam * q + mu * p
    else:
        stokes += 1e-8 * p * q

    a = BilinearForm(X, symmetric=False, condense=condense)
    a += stokes.Compile(**compile_opts) * dx
    a.Assemble()

    f = LinearForm(X)
    f += (f_vol * v).Compile(compile_flag, wait=wait_compile) * dx
    f.Assemble()

    if g_bnd_dom is not None:
        gfu.components[0].Set(g_bnd,
                              definedon=mesh.Boundaries(g_bnd_dom))
    else:
        gfu.vec.data[:] = 0.0

    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec
    if condense:
        r.data += a.harmonic_extension_trans * r
    gfu.vec.data += a.mat.Inverse(X.FreeDofs(condense), inverse=inverse) * r
    if condense:
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * r

    if lagr:
        vel, pre, lam = gfu.components
    else:
        vel, pre = gfu.components
    return vel, pre
