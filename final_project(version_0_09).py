#/Users/nicolinoprimavera

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import dolfin
import os
from fenics import *
import pandas as pd
def dolfinx(): # # need these packages if solving with dolfinx
    pass    
    #import dolfinx
    #from mpi4py import MPI
    #import ufl
    #from petsc4py import PETSc
    #import pyvista as pv

    # if using dolfinx
    #from dolfinx.fem import Constant, Function, functionspace, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
    #from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc, LinearProblem
    #from dolfinx.io import VTXWriter, XDMFFile
    #from dolfinx.mesh import create_rectangle, CellType
    #from dolfinx.plot import vtk_mesh
    #from basix.ufl import element
    #from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
    #                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym,
    #                 CellDiameter, Constant, as_vector)

    # Enable off-screen rendering
    #pv.OFF_SCREEN = True

# Check installation
print(f"\nChecking installation:")
#print(f"\nFEniCSx version (dolfinx): {dolfinx.__version__}")
#print(f"MPI initialized: {MPI.initialized()}")
#print(f"pyvista version: {pv.__version__}")
#print(f"UFL version: {ufl.__version__}")
print(f"FEniCS version (dolfin): {dolfin.__version__}")

"""Nicolino Primavera
FEM for Fluid Flow and FSI Interactions
Final Project
12/17/24

Problem II: Stokes Flow in a 2D Channel

Objective: compare the numerically predicted pressure gradient against anyalytical value assuming stokes flow

Stokes flow governing equations:
  - ρ(∂u/dt - f^b) = ∇ ∙ σ
  - ∇ ∙ u = 0 (incompressibility)
  - -∇p + μ∇^2u = 0     (∇^2u represents the Laplacian of velocity)

Stabilized Stokes Equations:
  - Petrov Galerkin method to enhance stability without losing consistency 
  - Galerkin formulation + add stabilization terms
  - stabilization allows using equal-order discretization for both velocity and pressure function spaces (ex. P1-P1, Q1-Q1)

Variables:
  - σ_ij = -p*δ_ij + μ(u_i,j + u_j,i) - Cauchy stress for the fluid
  - u - fluid velocity
  - p - fluid pressure
  - f^b - body force acting on the fluid
  - ρ - fluid density
  - μ - dynamic viscosity

Channel dimensions: 1cm (height) x 8cm (length)

Boundary Conditions:
  - Inflow: Dirichlet condition with a velocity profile u/U_∞ = y/H * (1 - y/H) where U_∞ = 1 cm/s is the velocity at the centerline
  - Outflow: Traction-free boundary condition
  - Top/Bottom faces: Fixed walls (no slip / no penetration condition, u = 0)

Physical Parameters:
  - Incompressible flow, no body force (f^b = 0)
  - Fluid Density (ρ): 1.0 g/cm^3
  - Fluid Viscosity (μ): 1.0 g/cm/s

The Reynolds number of the flow Re = (ρ)*(U_∞)*(H)/μ = 1, therefore, a Stokes flow is a reasonable assumption for the above configuration

Problems:
- (a) Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure. E.g., P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)
- (b) Simulate the above problem using inf-sup conforming finite elements. E.g., Q2Q1 discretization where the velocity functions are biquadratic (9-noded quadrilateral) and pressure is approximated using bilinear (4-noded quadrilateral) elements.
- (c) Plot profiles of velocity and streamlines at any axial cross-section.
- (d) Plot pressure along the centerline and compare the gradient against analytical expression assuming a fully developed parallel flow.
"""

# Starting program
print("\nWorking directory:", os.getcwd())
print("\nStarting Program...\n")

# Initialize channel dimensions
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
print(f"\nChannel dimensions: {L} cm x {H} cm\n")

# Fluid/Physical parameters
ρ = 1           # fluid density (g/cm^3)
rho = 1
μ = 1           # dynamic fluid viscosity (g/cm/s)
mu = 1
print(f"\nFluid density: ρ = {rho} g/cm^3 \nFluid viscosity: μ = {mu} g/cm/s\n")
U_max = 1       # fluid velocity (cm/s) at the centerline - inflow

# Incompressible flow
f_b = 0   # no body force
print(f"\nIncompressible flow = no body force (f_b = {f_b} N)")

# Reynolds number
Re = ρ * U_max * H / μ
print(f"\nReynolds number: Re = {Re} and therefore, a Stokes flow is a reasonable assumption\n")

# Define the run-time and temporal discretization
def simulation_params():
    t = 0
    T = 10
    num_steps = 500
    dt = T / num_steps

# P1P1 Simulation with Stabilization
print("\nSolving part (a): Running simulation with P1P1 linear elements (equal-order)...")

"""Part (a)
- Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure.
- Ex: P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)
"""

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

def dolfinx_mesh(): # if using dolfinx
    pass
    # Create a rectangular mesh with a specified number of elements
    #corner_points = [(0.0, 0.0), (8.0, 1.0)]
    #resolution = (nx, ny)  # Number of elements in x and y directions
    #mesh = create_rectangle(MPI.COMM_WORLD, corner_points, resolution)
    #mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [L, H]], [nx, ny], CellType.triangle)

print("\nFEM mesh created.")

# P1P1: Linear elements for both velocity and pressure
V = VectorFunctionSpace(mesh, "P", 1)  # Velocity space (vector field)
Q = FunctionSpace(mesh, "P", 1)        # Pressure space (scalar field)
element = MixedElement([VectorElement("P", mesh.ufl_cell(), 1), FiniteElement("P", mesh.ufl_cell(), 1)])    # Define mixed element for velocity (vector field) and pressure (scalar field)
W = FunctionSpace(mesh, element)    # Create the mixed function space using the mixed element

# Define trial and test functions for variational formulation
(u, p) = TrialFunctions(W)  # Trial functions for velocity (u) and pressure (p)
(v, q) = TestFunctions(W)  # Test functions for velocity (v) and pressure (q)

# If you want to test W function space 
w_test = Function(W)    # Create a zero function in the mixed space
u_mixed, p_mixed = w_test.split()   # Split into velocity and pressure components

def function_space_dolfinx():
    pass
    # Create two function spaces using the ufl element definitions as input - if using dolfinx
    #v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
    #s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    #V = functionspace(mesh, v_cg2)  # vector valued function space for the velocity - use piecewise quadratic elements for the velocity
    #P = functionspace(mesh, s_cg1)  # scalar valued function space for the pressure - use piecewise linear elements for the pressure

    # Create trial and test functions - since we have two different function spaces, we need to create two sets of trial and test functions
    #u = TrialFunction(W)
    #v = TestFunction(W)
    #p = TrialFunction(W)
    #q = TestFunction(W)

#print("Creation of function spaces complete.\n")
print("\nVelocity function space is a vector field (2D).")
print("Pressure function space is a scalar field (1D).\n")
#print("Trial and test functions created.")

# Maximum velocity at the centerline
U_max = 1.0 

def parabolic_inflow_dolfinx():
    pass
    # Define the parabolic inflow velocity profile - u/U_inf = y/H - y^2/H^2
    #U_in = Function(V)  # Define a velocity Function in the velocity function space - if using dolfinx

    # Define the parabolic inflow velocity profile - if using dolfinx
    #def parabolic_inflow(x):
    #    values = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=np.float64)
    #    values[0] = 4.0 * U_max * x[1] * (H - x[1]) / (H**2)  # x-component (parabolic profile)
    #    values[1] = 0.0  # y-component (no flow in the vertical direction)
    #    print(f"Parabolic inflow velocity in the x-direction:\n \n{values[0]}\n")
    #    print(f"Parabolic inflow velocity in the y-direction:\n \n{values[1]}\n")
    #    return values

    # Interpolate the velocity profile into the function space - if using dolfinx
    #U_in.interpolate(parabolic_inflow)

# Define the parabolic inflow velocity profile
U_inlet = Expression(("4.0 * U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=1.0, H=H, degree=2)  # U_max = centerline velocity
print(f"\nInflow velocity profile: {U_inlet}\n")

# Analytical solution
# Create a mesh grid for the whole domain
x_vals = np.linspace(0, L, 100) # L = 8
y_vals = np.linspace(0, H, 100) # H = 1
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the velocity components across the domain
U_x = 4.0 * U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in x-direction
U_y = np.zeros_like(Y)                    # Zero velocity in the y-direction
print("\nAnalytical solution:\n")
print(f"Parabolic inflow velocity in the x-direction:\n \n{U_x}\n")
print(f"Parabolic inflow velocity in the y-direction:\n \n{U_y}\n")

## Stokes Equations (weak/variational form) - for incompressible flow in a 2D domain

# Define the bilinear form for Stokes equations
a_stokes = (
    mu * inner(grad(u), grad(v)) * dx        # Viscous term
    - div(v) * p * dx                    #
    + div(u) * q * dx                      #
)
print(f"\nBilinear form of stokes equations:\n {a_stokes}")
# Define the linear form (forcing term)
f_b = Constant((0.0, 0.0))  # Zero body force for 2D flow 
L_stokes = dot(f_b, v) * dx  # linear form , ∇ ∙ u = 0 (incompressibility)
#L_stokes = dot(Constant((0.0, 0.0)), v) * dx  # linear form , ∇ ∙ u = 0 (incompressibility)
print(f"\nLinear form of stokes equations:\n {L_stokes}\n")

# PSPG stabilization parameter τ (tau)
h = CellDiameter(mesh)  # Define characteristic element size (h) using UFL's CellDiameter
tau = h**2 / (4.0 * mu) # stabilization parameter 
print(f"\nPSPG stabilization parameter: {tau}\n")

# Stokes equations (stabilized) - stabilization allows using equal-order discretization for both velocity and pressure function spaces (ex. P1-P1, Q1-Q1)
a_P1P1 = (    # bilinear form
    μ * inner(grad(u), grad(v)) * dx       # Viscous term
    - div(v) * p * dx                                  #
    + div(u) * q * dx                                  #
    + tau * inner(grad(p), grad(q)) * dx   # PSPG stabilization term
)
print(f"\nBilinear form of stokes equations with stabilization term: \n{a_P1P1}")
L_P1P1 = dot(f_b, v) * dx  # linear form , ∇ ∙ u = 0 (incompressibility)
print(f"\nLinear form of stokes equations with stabilization term: \n{L_P1P1}")

## Dirichlet Boundary Conditions
inflow_bc = DirichletBC(W.sub(0), U_inlet, "near(x[0], 0)")    # Inflow boundary: Parabolic velocity profile
walls_bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[1], 0 || near(x[1], {H}))".format(H=H))   # Walls (top and bottom): No-slip condition
outflow_bc = DirichletBC(W.sub(1), Constant(0.0), "near(x[0], {L})".format(L=L)) # Outflow boundary: pressure 
# Combine boundary conditions
bcs = [inflow_bc, walls_bc, outflow_bc]

# Error handling
#print(f"Inflow boundary condition: \n{bcu_inflow}\n")
#print(f"Walls boundary condition: \n{bcu_walls}\n")
#print(f"Combined boundary conditions: \n{bcs}\n")

def bcs_dolfinx(): # if using dolfinx
    pass
    # Fixed walls (no-slip / no penetration condition, u = 0): set u = 0 at the walls of the channel (H = 0 and H = 1)  - if using dolfinx
    #def walls(x):
    #    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], H))
    #wall_dofs = locate_dofs_geometrical(V, walls)
    #u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    #bc_noslip = dirichletbc(u_noslip, wall_dofs, V)

    # Inflow: set p = 1 at the inflow (x = 0)
    #def inflow(x):
    #    return np.isclose(x[0], 0)
    #inflow_dofs = locate_dofs_geometrical(P, inflow)
    #bc_inflow = dirichletbc(PETSc.ScalarType(1), inflow_dofs, P)

    # Outflow (Traction-free boundary condition): set p = 1 at the outflow (x = 8)
    #def outflow(x):
    #    return np.isclose(x[0], L)
    #outflow_dofs = locate_dofs_geometrical(P, outflow)
    #bc_outflow = dirichletbc(PETSc.ScalarType(1), outflow_dofs, P)

    # collect the BCs for the velocity and pressure in lists
    #bcu = [bc_noslip]
    #bcp = [bc_inflow, bc_outflow]

print("\nDirichlet boundary conditions defined.\n")

# P1P1 Solver
print("\nSolving the P1P1 solution...\n")
w_P1P1 = Function(W) # Create a function to store the solution
solve(a_P1 == L_P1, w_P1P1, bcs) # Solve the linear system
(velocity_solution, pressure_solution) = w_P1P1.split()  # Extract velocity and pressure solutions

# Velocity solutions
velocity = velocity_solution.compute_vertex_values(mesh)
print(f"\nVelocity: \n{velocity}\n")    # error handling
if not np.isfinite(velocity).all():
    raise ValueError("\nVelocity contains non-finite values (NaN or Inf).")

# Pressure solutions 
pressure = pressure_solution.compute_vertex_values(mesh)
print(f"\nPressure: \n{pressure}\n")    # error handling
if not np.isfinite(pressure).all():
    raise ValueError("\nPressure contains non-finite values (NaN or Inf).")

# Compute the numerical pressure gradient along the centerline
print("\nThe pressure gradient provides the slope of the pressure curve.")
x_coords = np.linspace(0, L, 100)  # Sample x-coordinates along the channel
pressure_gradient = (pressure_solution(Point(x_coords[-1], H/2)) - pressure_solution(Point(x_coords[0], H/2))) / L
print(f"Numerical Pressure Gradient: {pressure_gradient:.4f}")

# Compute analytical pressure gradient
#dp/dx = -2 * μ * U_max / (H**2)
analytical_pressure_gradient = -2 * μ * U_max / (H**2)
print(f"Analytical Pressure Gradient = dp/dx = {analytical_pressure_gradient}")

# Pressure - example of an analytical solution
p_o = 0   # initial pressure
p_f = -16 # final pressure (analytical solution)
print(f"Pressure Analytical Solution:\n    - If the initial pressure: p_o = {p_o}psi, then the final pressure: p_f = {p_f}psi")
print(f"    - This is because of the formula:  p = p_o - 2 * L\n")

# Relative error between numerical and analytical pressure gradients
relative_error = abs((pressure_gradient - analytical_pressure_gradient) / analytical_pressure_gradient) * 100
print(f"Relative Error: {relative_error:.2f}%")

# Plot velocity field (magnitude)
print(f"\nVelocity solution: \n{velocity_solution}")
plt.figure()
plot_u = plot(velocity_solution, title="Velocity Field", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_u, label="Velocity value")  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
#plt.show()

# Plot pressure field
print(f"\nPressure solution: \n{pressure_solution}\n")
plt.figure()
plot_p = plot(pressure_solution, title="Pressure Field", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_p, label="Pressure value")  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
#plt.show()

"""Part b

- Simulate the above problem using inf-sup conforming finite elements. 
- E.g., Q2Q1 discretization where the velocity functions are biquadratic (9-noded quadrilateral) and pressure is approximated using bilinear (4-noded quadrilateral) elements.

"""

# Q2Q1 Solver
print("\nSolving the Q2Q1 solution...\n")
w_Q2Q1 = Function(W) # Create a function to store the solution
solve(a_stokes == L_stokes, w_Q2Q1, bcs) # Solve the linear system
(velocity_solution_Q2Q1, pressure_solution_Q2Q1) = w_Q2Q1.split()  # Extract velocity and pressure solutions

# Velocity solutions
velocity_Q2Q1 = velocity_solution_Q2Q1.compute_vertex_values(mesh)
print(f"\nVelocity: \n{velocity_Q2Q1}\n")    # error handling
if not np.isfinite(velocity_Q2Q1).all():
    raise ValueError("\nVelocity contains non-finite values (NaN or Inf).\n")

# Pressure solutions 
pressure_Q2Q1 = pressure_solution_Q2Q1.compute_vertex_values(mesh)
print(f"\nPressure: \n{pressure_Q2Q1}\n")    # error handling
if not np.isfinite(pressure_Q2Q1).all():
    raise ValueError("\nPressure contains non-finite values (NaN or Inf).\n")

# Compute the numerical pressure gradient along the centerline
#print("\nThe pressure gradient provides the slope of the pressure curve.")
x_coords_Q2Q1 = np.linspace(0, L, 100)  # Sample x-coordinates along the channel
pressure_gradient_Q2Q1 = (pressure_solution_Q2Q1(Point(x_coords[-1], H/2)) - pressure_solution_Q2Q1(Point(x_coords[0], H/2))) / L
print(f"Numerical Pressure Gradient: {pressure_gradient_Q2Q1:.4f}")