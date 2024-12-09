
## Generic UFL input for the mixed stokes problem
# Define function spaces
V = VectorFunctionSpace(mesh, U_element, U_order)
Q = FunctionSpace(mesh, P_element, P_order) W=V*Q
# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
# Define the variational problems
a = inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx L = inner(f, v)*dx


## Generic UFL input for defining the MINI element velocity space 
# Define function spaces
P = VectorFunctionSpace(mesh, "Lagrange", U_order)
B = VectorFunctionSpace(mesh, "Bubble", U_order + 2)
V=P+B


## UFL code to add stabilization to the mixed method code from the chart: Element variables defining the different mixed methods 
# Sample parameters for pressure stabilization
h = CellSize(mesh)
beta = 0.2
delta = beta*h**2
# The additional pressure stabilization terms
a += delta*inner(grad(p), grad(q))*dx
L += delta*inner(f, grad(q))*dx


## DOLFIN code for defining the Scott-Vogelius method 
# Define function space
V = VectorFunctionSpace(mesh, "Lagrange", U_order)
# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
# Define auxiliary function and parameters
w = Function(V);
rho = 1.0e3
r = -rho
# Define the variational problem
a = inner(grad(u), grad(v))*dx + r*div(u)*div(v)*dx
L = inner(f, v)*dx + inner(div(w), div(v))*dx
U = Function(V)
pde = LinearVariationalProblem(a, L, U, bc0)
solver = LinearVariationalSolver(pde);
# Iterate to fix point
iters = 0; max_iters = 100; U_m_u = 1 
while iters < max_iters and U_m_u > 1e-8:
    solver.solve()
    w.vector().axpy(rho, U.vector())
    if iters != 0:
        U_m_u = (U.vector() - u_old_vec).norm("l2")
    u_old_vec = U.vector().copy()
    iters += 1


## DOLFIN code for defining the test domain 
# Define the boundary domains
class NoSlipDomain(SubDomain):
    def inside(self, x, on_boundary): 
        return on_boundary
    
class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Define mesh
mesh = UnitSquare(h_num, h_num, "crossed")

# Instantiate the boundary conditions, set the
# velocity dof values to the exact solution, and
# pinpoint the pressure.
noslip_domain = NoSlipDomain()
noslip = Expression(("sin(4*pi*x[0])*cos(4*pi*x[1])", 
                     "-cos(4*pi*x[0])*sin(4*pi*x[1])"))
pinpoint = PinPoint()
pin_val = Expression("pi*cos(4*pi*x[0])*cos(4*pi*x[1])")
bc0 = DirichletBC(W.sub(0), noslip, noslip_domain)
bc1 = DirichletBC(W.sub(1), pin_val, pinpoint, "pointwise")
bc = [bc0, bc1]

# Define the RHS
f = Expression(("28*pi**2*sin(4*pi*x[0])"\
                "cos(4*pi*x[1])", 
                "-36*pi**2*cos(4*pi*x[0])*sin(4*pi*x[1])"))


## DOLFIN code for defining the lid-driven cavity domain
# Define the boundary domains
class NoSlipDomain(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 1.0 - DOLFIN_EPS

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] > 1.0 - DOLFIN_EPS

class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

# Define mesh
mesh = UnitSquare(h_num, h_num, "crossed")

# Instantiate the boundary conditions
noslip_domain = NoSlipDomain()
noslip_val = Constant((0.0, 0.0))
top_domain = Top()
top_val = Expression(("x[0]*(1.0 - x[0])", "0.0")) pinpoint = PinPoint()
pin_val = Constant(0.0)

# Define the RHS
f = Constant((0.0, 0.0))


## DOLFIN code for computing the error in the L^2 norm. The exact solution is interpolated using 10th order Lagrange polynomials on cells 
# Define a high order approximation to the exact solution
u_ex = Expression(("sin(4*pi*x[0])*cos(4*pi*x[1])", 
                    "-cos(4*pi*x[0])*sin(4*pi*x[1])"),
                    element=VectorElement("Lagrange", triangle, 
                            10))
p_ex = Expression("pi*cos(4*pi*x[0])*cos(4*pi*x[1])",
                element=FiniteElement("Lagrange", triangle, 
                    10))

# Define the L2 error norm
M_u = inner((u_ex - u),(u_ex - u))*dx
M_p = (p_ex - p)*(p_ex - p)*dx

# Compute the integral
u_err = assemble(M_u, mesh=mesh)
p_err = assemble(M_p, mesh=mesh)

# Compute L2 error of the divergence
M_div = div(u)*div(u)*dx
div_err = assemble(M_div, mesh=mesh)