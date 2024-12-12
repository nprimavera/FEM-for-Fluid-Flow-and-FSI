#/Users/nicolinoprimavera

# Install necessary packages 
!pip install -q condacolab
import condacolab
condacolab.install()

!conda install -c conda-forge fenics-dolfinx -y

!pip install fenics-dolfinx fenics-ufl petsc4py mpi4py pyvista

import dolfinx
from mpi4py import MPI
import ufl
import numpy as np
from petsc4py import PETSc

!pip install pyvista pyvistaqt
!apt-get install -y xvfb libgl1-mesa-glx

import pyvista as pv

# Enable off-screen rendering
pv.OFF_SCREEN = True

# Check installation
print(f"\nChecking installation:")
print(f"\nFEniCSx version: {dolfinx.__version__}")
print(f"MPI initialized: {MPI.Is_initialized()}")
print(f"pyvista version: {pyvista.Is_initialized()}")

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista
import os

from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)

# Initialize channel parameters
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
print(f"Channel dimensions: {L} cm x {H} cm")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
rho = 1
μ = 1           # fluid viscosity (g/cm/s)
mu = 1
print(f"\nFluid density: ρ = {rho} g/cm^3 \nFluid viscosity: μ = {mu} g/cm/s")
U_infinity = 1  # fluid velocity (cm/s) at the centerline - inflow

# Incompressible flow
f_b = 0   # no body force
print(f"\nIncompressible flow = no body force (f_b = {f_b} N)")
f_b = Constant((0.0, 0.0))  # Zero body force for 2D flow

# Reynolds number
Re = ρ * U_infinity * H / μ
print(f"\nReynolds number: Re = {Re} and therefore, a Stokes flow is a reasonable assumption")

# Pressure - example of an analytical solution
p_o = 0   # initial pressure
p_f = -16 # final pressure (analytical solution)
print(f"\nAnalytical Solution:\n  If the initial pressure: p_o = {p_o}psi, then the final pressure: p_f = {p_f}psi")
print(f"  This is because of the formula:  p = p_o - 2 * L\n")

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)  # 2D mesh

# Define the run-time and temporal discretization
t = 0
T = 10
num_steps = 500
dt = T / num_steps

# Create two function spaces using the ufl element definitions as input
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)  # vector valued function space for the velocity - use piecewise quadratic elements for the velocity
Q = functionspace(mesh, s_cg1)  # scalar valued function space for the pressure - use piecewise linear elements for the pressure

# Create trial and test functions - since we have two different function spaces, we need to create two sets of trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

## Dirichlet Boundary Conditions

# set u = 0 at the walls of the channel, that is at H = 0 and H = 1
def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

wall_dofs = locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_noslip = dirichletbc(u_noslip, wall_dofs, V)

# set p = 1 at the inflow (x = 0)
def inflow(x):
    return np.isclose(x[0], 0)

inflow_dofs = locate_dofs_geometrical(Q, inflow)
bc_inflow = dirichletbc(PETSc.ScalarType(1), inflow_dofs, Q)

# set p = 1 at the outflow (x = 8) - results in a pressure gradient that will accelerate the flow from the initial state with zero velocity
def outflow(x):
    return np.isclose(x[0], 8)

outflow_dofs = locate_dofs_geometrical(Q, outflow)
bc_outflow = dirichletbc(PETSc.ScalarType(1), outflow_dofs, Q)

# collect the BCs for the velocity and pressure in lists
bcu = [bc_noslip]
bcp = [bc_inflow, bc_outflow]

## Definition of the three variational forms - one for each step in the IPCS scheme (Incremental Pressure Correction Scheme)
u_n = Function(V)
u_n.name = "u_n"
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant(mesh, PETSc.ScalarType((0, 0)))
k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(1))
rho = Constant(mesh, PETSc.ScalarType(1))

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

# Define the variational problem for the first step - Predictor Step (Momentum Equation)
p_n = Function(Q)
p_n.name = "p_n"
F1 = rho * dot((u - u_n) / k, v) * dx
F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
F1 += inner(sigma(U, p_n), epsilon(v)) * dx
F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
F1 -= dot(f, v) * dx
a1 = form(lhs(F1))
L1 = form(rhs(F1))

# Analytical solution
A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(L1)

# Define variational problem for step 2 - Pressure Correction Step (Poisson Equation for Pressure)
u_ = Function(V)
a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# Define variational problem for step 3 - Velocity Correction Step
p_ = Function(Q)
a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

# Create a solver
"""
For the tentative velocity step and pressure correction step, we will use the Stabilized version of BiConjugate Gradient to solve the linear system,
and using algebraic multigrid for preconditioning. For the last step, the velocity update, we use a conjugate gradient method with successive over
relaxation, Gauss Seidel (SOR) preconditioning.
"""
# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# Prepare output files for the velocity and pressure data, and write the mesh and initial conditions to file
from pathlib import Path
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(mesh.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
vtx_p = VTXWriter(mesh.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
vtx_u.write(t)
vtx_p.write(t)

# Interpolate the analytical solution into our function-space and create a variational formulation for the L^2 error
def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values

u_ex = Function(V)
u_ex.interpolate(u_exact)

L2_error = form(dot(u_ - u_ex, u_ - u_ex) * dx)

# Create the loop over time
for i in range(num_steps):
    # Update current time step
    t += dt

    # Step 1: Tentative veolcity step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.x.petsc_vec)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.x.petsc_vec)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.x.petsc_vec)
    u_.x.scatter_forward()
    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # Write solutions to file
    vtx_u.write(t)
    vtx_p.write(t)

    # Compute error at current time-step
    error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    error_max = mesh.comm.allreduce(np.max(u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX)
    # Print error only every 20th step and at the last step
    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")
# Close xmdf file
vtx_u.close()
vtx_p.close()
b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

# Visualization
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(u_n)] = u_n.x.array.real.reshape((geometry.shape[0], len(u_n)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)

# Create a pyvista-grid for the mesh
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")




# Import necessary libraries
!pip install fenics-dolfinx fenics-ufl petsc4py mpi4py pyvista

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_rectangle
from dolfinx.io import XDMFFile
from dolfinx.fem import (Function, FunctionSpace, Constant, dirichletbc, form, locate_dofs_geometrical, assemble_matrix, assemble_vector, apply_lifting, set_bc)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.cpp.mesh import CellType
import ufl
import numpy as np

# Channel dimensions and mesh
L, H = 8.0, 1.0  # Length and Height of the channel in cm
nx, ny = 80, 10  # Mesh resolution
mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [L, H]], [nx, ny], CellType.triangle)

# Define function spaces for velocity (vector) and pressure (scalar)
V = FunctionSpace(mesh, ("CG", 2))  # Quadratic elements for velocity
Q = FunctionSpace(mesh, ("CG", 1))  # Linear elements for pressure

# Boundary conditions
def walls(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], H))

def inflow(x):
    return np.isclose(x[0], 0.0)

def outflow(x):
    return np.isclose(x[0], L)

# No-slip boundary condition for velocity on walls
wall_bc = dirichletbc(PETSc.ScalarType(0), locate_dofs_geometrical(V, walls), V)

# Inflow parabolic velocity profile
inflow_profile = Function(V)
inflow_profile.interpolate(lambda x: np.vstack((4.0 * x[1] * (H - x[1]) / H**2, np.zeros_like(x[1]))))
inflow_bc = dirichletbc(inflow_profile, locate_dofs_geometrical(V, inflow), V)

# Collect velocity boundary conditions
bcu = [wall_bc, inflow_bc]

# Pressure boundary condition at outflow (zero mean)
outflow_bc = dirichletbc(PETSc.ScalarType(0), locate_dofs_geometrical(Q, outflow), Q)
bcp = [outflow_bc]

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

# Define viscosity and body force
mu = Constant(mesh, PETSc.ScalarType(1.0))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1.0))  # Density
f = Constant(mesh, PETSc.ScalarType((0.0, 0.0)))  # No body force

# Define the weak form of the Stokes equations
# Momentum equation
a_momentum = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.div(v) * p * ufl.dx
L_momentum = ufl.inner(f, v) * ufl.dx

# Continuity equation
a_continuity = ufl.div(u) * q * ufl.dx
L_continuity = Constant(mesh, PETSc.ScalarType(0)) * q * ufl.dx

# Assemble and solve using a monolithic solver
a = a_momentum + a_continuity
L = L_momentum + L_continuity
problem = LinearProblem(a, L, bcs=bcu + bcp)
solution = problem.solve()

# Split solution into velocity and pressure
u_h = solution.sub(0)
p_h = solution.sub(1)

# Save results to file
with XDMFFile(MPI.COMM_WORLD, "stokes_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h, "Velocity")
    xdmf.write_function(p_h, "Pressure")

# Plot velocity and pressure profiles
import pyvista as pv
pv.start_xvfb()  # Enable off-screen rendering
mesh_topology, cell_types, geometry = ufl.plot.vtk_mesh(mesh)
grid = pv.UnstructuredGrid(mesh_topology, cell_types, geometry)
grid["Velocity"] = u_h.x.array.reshape(-1, 2)
grid["Pressure"] = p_h.x.array
grid.plot(scalars="Velocity", vector_mode="Magnitude")
grid.plot(scalars="Pressure")






from dolfinx.mesh import create_rectangle
from mpi4py import MPI
from dolfinx.fem import (FunctionSpace, Function, dirichletbc, form, assemble_matrix, assemble_vector, apply_lifting, set_bc, Constant)
from ufl import (TrialFunction, TestFunction, FacetNormal, dx, ds, dot, grad, div, sym, inner, Identity)

# Create a 2D rectangular mesh
L, H = 8.0, 1.0  # Length and height of the channel
nx, ny = 80, 10  # Mesh resolution
mesh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [nx, ny], cell_type="triangle")

# Create function spaces for velocity and pressure (P1P1 elements)
V = FunctionSpace(mesh, ("CG", 1))  # Linear elements for velocity
Q = FunctionSpace(mesh, ("CG", 1))  # Linear elements for pressure

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Fluid parameters
mu = Constant(mesh, 1.0)  # Dynamic viscosity
rho = Constant(mesh, 1.0)  # Density
f = Constant(mesh, (0.0, 0.0))  # Body force

# Define stabilization parameter τ (tau)
h = mesh.topology.cell_diameter  # Characteristic element size
tau = h**2 / (4 * mu)

# Stabilization terms for PSPG
pspg_stab = tau * inner(grad(p), grad(q)) * dx

# Weak form for the momentum equation (with PSPG)
momentum = (rho * inner(grad(u) * u, v) * dx +
            2 * mu * inner(sym(grad(u)), sym(grad(v))) * dx -
            inner(p, div(v)) * dx +
            inner(f, v) * dx)

# Weak form for the continuity equation (with PSPG)
continuity = (div(u) * q * dx + pspg_stab)

# Combine the momentum and continuity equations
F = momentum + continuity
a = form(F)

# Boundary conditions
def inflow(x):
    return np.isclose(x[0], 0)

def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], H))

def outflow(x):
    return np.isclose(x[0], L)

# No-slip condition on walls
wall_bc = dirichletbc(PETSc.ScalarType(0), locate_dofs_geometrical(V, walls), V)

# Inflow parabolic velocity profile
inflow_profile = Function(V)
inflow_profile.interpolate(lambda x: 4.0 * x[1] * (H - x[1]) / H**2)
inflow_bc = dirichletbc(inflow_profile, locate_dofs_geometrical(V, inflow), V)

# Collect boundary conditions
bcu = [wall_bc, inflow_bc]
bcp = []  # No explicit pressure Dirichlet BC; handled implicitly via stabilization

# Assemble system and solve
A = assemble_matrix(a, bcs=bcu + bcp)
A.assemble()
b = create_vector(F)
assemble_vector(b, F)
apply_lifting(b, [a], bcs=[bcu, bcp])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcu)

# Create solution vector
u_sol = Function(V)
p_sol = Function(Q)

# Solve the linear system
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.MINRES)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
solver.solve(b, u_sol.vector)

# Output results to file
from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_sol)

with XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w") as xdmf:
    xdmf.write_function(p_sol)
