# Install fenics
def fenics():
    try:
        import dolfin
    except ImportError:
        # Install FEniCS if not installed
        !wget "https://fem-on-colab.github.io/releases/fenics-install-real.sh" -O "/tmp/fenics-install.sh"
        !bash "/tmp/fenics-install.sh"

    # Now try importing again
    import dolfin
fenics()    # Execute function

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import pyplot as plt
from matplotlib import cm
from fenics import *

# Generic UFL input for the mixed stokes problem
# Define function spaces
V = VectorFunctionSpace(mesh, U_element, U_order)
Q = FunctionSpace(mesh, P_element, P_order) W=V*Q
# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
# Define the variational problems
a = inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx L = inner(f, v)*dx

# Generic UFL input for defining the MINI element velocity space 
# Define function spaces
P = VectorFunctionSpace(mesh, "Lagrange", U_order)
B = VectorFunctionSpace(mesh, "Bubble", U_order + 2)
V=P+B

# UFL code to add stabilization to the mixed method code from the chart: Element variables defining the different mixed methods 
# Sample parameters for pressure stabilization
h = CellSize(mesh)
beta = 0.2
delta = beta*h**2
# The additional pressure stabilization terms
a += delta*inner(grad(p), grad(q))*dx
L += delta*inner(f, grad(q))*dx

# DOLFIN code for defining the Scott-Vogelius method 
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

# DOLFIN code for defining the test domain 
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

