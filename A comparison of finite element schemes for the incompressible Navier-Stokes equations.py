# Incompressible Navier-Stokes equations with unit fluid density 
# u ̇ + u·∇u − ∇·σ = f
# ∇·u = 0 

# Cauchy stress tensor which for a Newtonian fluid: σ(u, p) = 2νε(u) − pI.

# ε(u)= 1/2 (∇u+∇u⊤) - symmetric gradient 
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u, p, nu):
    return 2*nu*epsilon(u) - p*Identity(u.cell().d)


## Implementation of variational forms for the Chorin solver 
# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx \
    + inner(dot(u0, nabla_grad(u0)), v)*dx \
    + nu*inner(nabla_grad(u), nabla_grad(v))*dx \
    - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Poisson problem for the pressure
a2 = inner(nabla_grad(p), nabla_grad(q))*dx
L2 = -(1/k)*nabla_div(us)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(us, v)*dx - k*inner(nabla_grad(p1), v)*dx


## Implementation of variational forms for the IPCS solver. The flag beta = 1 is set to zero in the case when periodic boundary conditions are used.
# Tentative velocity step
U = 0.5*(u0 + u)
F1 = (1/k)*inner(u - u0, v)*dx \
    + inner(dot(u0, nabla_grad(u0)), v)*dx \
    + inner(sigma(U, p0, nu), epsilon(v))*dx \
    + inner(p0*n, v)*ds \
    - beta*nu*inner(dot(n, nabla_grad(U).T), v)*ds \
    - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure correction
a2 = inner(nabla_grad(p), nabla_grad(q))*dx
L2 = inner(nabla_grad(p0), nabla_grad(q))*dx \
    - (1.0/k)*nabla_div(u1)*q*dx

# Velocity correction
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(nabla_grad(p1 - p0), v)*dx


## Implementation of vari- ational forms for the CSS solver(s). The flag beta = 1 is set to zero in the case of periodic boundary conditions.
# Tentative pressure
if self.order == 1: 
    ps = p1
else:
    ps = 2*p1 - p0

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx \
   + inner(dot(u0, nabla_grad(u0)), v)*dx \
    + inner(sigma(u, ps, nu), epsilon(v))*dx \
    - beta*nu*inner(dot(n, nabla_grad(u).T), v)*ds \
    + inner(pbar*n, v)*ds \
    - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure correction
a2 = inner(nabla_grad(p), nabla_grad(q))*dx
L2 = (1/k)*inner(u1 - u0, nabla_grad(q))*dx \
   - (1/k)*inner(u1 - u0, q*n)*ds

# Pressure update
a3 = p*q*dx
L3 = p1*q*dx + psi*q*dx - nu*nabla_div(u1)*q*dx


## Implementation of variational forms for the G2 solver 
# Velocity system
U = 0.5*(u0 + u)
P = p1
Fv = (1/k)*inner(u - u0, v)*dx \
    + inner(dot(W, nabla_grad(U)), v)*dx \
    + inner(sigma(U, P, nu), epsilon(v))*dx \
    - beta*nu*inner(dot(n, nabla_grad(U).T), v)*ds \ + inner(pbar*n, v)*ds \
    - inner(f, v)*dx \
    + d1*inner(dot(W, nabla_grad(U)), \
           dot(W, nabla_grad(v)))*dx \
    + d2*nabla_div(U)*nabla_div(v)*dx
av = lhs(Fv)
Lv = rhs(Fv)

# Pressure system
ap = inner(nabla_grad(p), nabla_grad(q))*dx
Lp = -(1/d1)*nabla_div(u1)*q*dx

# Projection of velocity
aw = inner(w, z)*dx
Lw = inner(u1, z)*dx


## Implementation of variational forms for the GRPC solver 
# Velocity and pressure residuals
U = 0.5*(u0 + u1)
P = p01
Ru = inner(u1 - u0, v)*dx \
   + k*inner(dot(U, nabla_grad(U)), v)*dx \
    + k*inner(sigma(U, P, nu), epsilon(v))*dx \
    - beta*k*nu*inner(dot(n, nabla_grad(U).T), v)*ds \
    + k*inner(pbar*n, v)*ds \
    - k*inner(f, v)*dx
Rp = k*nabla_div(U)*q*dx


## Implementation of velocity boundary conditions for the driven cavity test problem
class BoundaryValue(Expression):
    def eval(self, values, x):
        if x[0] > DOLFIN_EPS and \
            x[0] < 1.0 - DOLFIN_EPS and \ x[1] > 1.0 - DOLFIN_EPS:
            values[0] = 1.0
            values[1] = 0.0
        else:
            values[0] = 0.0
            values[1] = 0.0


## Computing the stream function in DOLFIN
# Define variational problem
psi = TrialFunction(V)
q   = TestFunction(V)
a   = dot(nabla_grad(psi), nabla_grad(q))*dx
L   = dot(u[1].dx(0) - u[0].dx(1), q)*dx

# Define boundary condition
g  = Constant(0)
bc = DirichletBC(V, g, DomainBoundary())

# Compute solution
psi = Function(V)
solve(a == L, psi, bc)


## Implementation of periodic boundary conditions for the Taylor–Green vortex test problem
class PeriodicBoundaryX(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < (-1.0 + DOLFIN_EPS) and \
               x[0] > (-1.0 - DOLFIN_EPS) and \ 
               on_boundary

    def map(self, x, y):
        y[0] = x[0] - 2.0
        y[1] = x[1]

class PeriodicBoundaryY(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < (-1.0 + DOLFIN_EPS) and \
                x[1] > (-1.0 - DOLFIN_EPS) and \ 
                    on_boundary

def map(self, x, y): 
    y[0] = x[0]
    y[1] = x[1] - 2.0