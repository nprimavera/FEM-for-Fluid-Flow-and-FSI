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

# P1P1 Simulation with Stabilization
print("\nSolving part (a): Running simulation with P1P1 linear elements (equal-order)...")

"""Part (a)
- Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure.
- Ex: P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)
- Parts (c) and (d) plots are included
"""

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

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

#print("Creation of function spaces complete.\n")
print("\nVelocity function space is a vector field (2D).")
print("Pressure function space is a scalar field (1D).\n")
#print("Trial and test functions created.")

# Define the parabolic inflow velocity profile
U_max = 1.0 # Maximum velocity at the centerline
U_inlet = Expression(("U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=U_max, H=H, degree=2)  # U_max = centerline velocity
print(f"\nInflow velocity profile (analytical solution): {U_inlet}")

# Analytical solution
# Create a mesh grid for the whole domain
x_vals = np.linspace(0, L, 100) # L = 8
y_vals = np.linspace(0, H, 100) # H = 1
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the velocity components across the domain
U_x = U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in x-direction
U_y = np.zeros_like(Y)                    # Zero velocity in the y-direction
print("\nAnalytical solution for inflow velocity profile:\n")
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
stabilization = tau * inner(grad(p), grad(q)) * dx  # stabilization term 
print(f"\nPSPG stabilization parameter tau: {tau}")
print(f"Stabilization: {stabilization}\n")

# Stokes equations (stabilized) - stabilization allows using equal-order discretization for both velocity and pressure function spaces (ex. P1-P1, Q1-Q1)
a_P1P1 = (    # bilinear form
    μ * inner(grad(u), grad(v)) * dx       # Viscous term
    - div(v) * p * dx                                  #
    + div(u) * q * dx                                  #
    + tau * inner(grad(p), grad(q)) * dx   # PSPG stabilization term
)
print(f"\nBilinear form of stokes equations with stabilization term: \n{a_P1P1}")
L_P1P1 = dot(f_b, v) * dx  # linear form , ∇ ∙ u = 0 (incompressibility)
#L_P1P1 += stabilization
print(f"\nLinear form of stokes equations with stabilization term: \n{L_P1P1}")

## Dirichlet Boundary Conditions
inflow_bc = DirichletBC(W.sub(0), U_inlet, "near(x[0], 0)")    # Inflow boundary: Parabolic velocity profile
walls_bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[1], 0 || near(x[1], {H}))".format(H=H))   # Walls (top and bottom): No-slip condition
#walls_bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[1], 0) || near(x[1], H)")
outflow_bc = DirichletBC(W.sub(1), Constant(0.0), "near(x[0], {L})".format(L=L)) # Outflow boundary: Pressure 
bcs = [inflow_bc, walls_bc, outflow_bc] # Combine boundary conditions

# Error handling
#print(f"Inflow boundary condition: \n{bcu_inflow}\n")
#print(f"Walls boundary condition: \n{bcu_walls}\n")
#print(f"Combined boundary conditions: \n{bcs}\n")

print("\nDirichlet boundary conditions defined.")

# P1P1 Solver
print("\nSolving P1P1 solution...\n")
w_P1P1 = Function(W) # Create a function to store the solution
solve(a_P1P1 == L_P1P1, w_P1P1, bcs) # Solve the linear system
(velocity_solution_P1P1, pressure_solution_P1P1) = w_P1P1.split()  # Extract velocity and pressure solutions

# P1P1 Velocity solutions
velocity_magnitude = sqrt(dot(velocity_solution_P1P1, velocity_solution_P1P1))
velocity_P1P1 = velocity_solution_P1P1.compute_vertex_values(mesh)
print(f"\nVelocity: \n{velocity_P1P1}")    # error handling
if not np.isfinite(velocity_P1P1).all():
    raise ValueError("\nVelocity contains non-finite values (NaN or Inf).")

# P1P1 Pressure solutions 
pressure_P1P1 = pressure_solution_P1P1.compute_vertex_values(mesh)
print(f"\nPressure: \n{pressure_P1P1}\n")    # error handling
if not np.isfinite(pressure_P1P1).all():
    raise ValueError("\nPressure contains non-finite values (NaN or Inf).")

# P1P1 - Compute the numerical pressure gradient along the centerline
print("\nThe pressure gradient provides the slope of the pressure curve.")
# Pressure - example of an analytical solution
p_o = 16   # initial pressure
p_f = 0 # final pressure (analytical solution)
print(f"Pressure Gradient Analytical Solution:\n    - If the initial pressure: p_o = {p_o}psi, then the final pressure: p_f = {p_f}psi")
print(f"    - This is because of the formula:  p = p_o - 2 * L\n")
# Compute analytical pressure gradient
#dp/dx = -2 * μ * U_max / (H**2)
analytical_pressure_gradient = -2 * μ * U_max / (H**2)
print(f"Analytical Pressure Gradient = dp/dx = {analytical_pressure_gradient}")
x_coords = np.linspace(0, L, 100)  # Sample x-coordinates along the channel
pressure_gradient_P1P1 = (pressure_solution_P1P1(Point(x_coords[-1], H/2)) - pressure_solution_P1P1(Point(x_coords[0], H/2))) / L
print(f"Numerical Pressure Gradient: {pressure_gradient_P1P1:.4f}")

# Relative error between numerical and analytical pressure gradients
relative_error_P1P1 = abs((pressure_gradient_P1P1 - analytical_pressure_gradient) / analytical_pressure_gradient) * 100
print(f"Relative Error: {relative_error_P1P1:.4f}%\n")

# Relative error between numerical and analytical velocity gradients
error = errornorm(U_inlet, velocity_solution_P1P1, norm_type="L2") * 100
print(F"Relative error for velocity: {error:.4f}%\n")

# Create folder to save plots 
save_folder = "/Users/nicolinoprimavera/Desktop/Columbia University/Finite Element Method for Fluid Flow and Fluid-Structure Interactions/Final Project/Plots" 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)  

# P1P1 - Plot velocity field (magnitude)
print(f"\nVelocity solution plot: \n{velocity_solution_P1P1}")
plt.figure()
plot_u_P1P1 = plot(velocity_solution_P1P1*4, title="Velocity Field for P1P1", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_u_P1P1, label="Velocity value")  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
# Save the plot 
filename = f"Velocity Field for P1P1.png"
save_path = os.path.join(save_folder, filename)
plt.savefig(save_path)
print(f"\nGraph saved as {save_path}.\n")
plt.show()

# P1P1 - Plot pressure field
print(f"\nPressure solution plot: \n{pressure_solution_P1P1}\n")
plt.figure()
plot_p_P1P1 = plot(pressure_solution_P1P1, title="Pressure Field for P1P1", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_p_P1P1, label="Pressure value")  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
# Save the plot 
filename = f"Pressure Field for P1P1.png"
save_path = os.path.join(save_folder, filename)
plt.savefig(save_path)
print(f"\nGraph saved as {save_path}.\n")
plt.show()

## Plot the pressure along the centerline 
print(f"Pressure along the centerline plot:\n")
# Initialize arrays to store numerical and analytical pressure values
numerical_pressure_P1P1 = []
analytical_pressure_P1P1 = []

# Compute pressure values along the centerline
for x in x_coords:
    numerical_pressure_P1P1.append(pressure_solution_P1P1(Point(x, H/2)))  # Numerical pressure
    analytical_pressure_P1P1.append(p_o + analytical_pressure_gradient * x)  # Analytical pressure

# Plot the pressure distributions
plt.figure(figsize=(10, 6))
plt.plot(x_coords, numerical_pressure_P1P1, label="Numerical Pressure (P1P1)", color="blue", linewidth=2)
plt.plot(x_coords, analytical_pressure_P1P1, label="Analytical Pressure", color="red", linestyle="--", linewidth=2)
plt.xlabel("x (Length along channel)", fontsize=12)
plt.ylabel("Pressure (psi)", fontsize=12)
plt.title("Pressure Distribution Along the Centerline of the Channel for P1P1", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
# Save the plot 
filename = f"Pressure Distribution Along the Centerline of the Channel for P1P1.png"
save_path = os.path.join(save_folder, filename)
plt.savefig(save_path)
print(f"\nGraph saved as {save_path}.\n")
plt.show()

"""Part b
- Simulate the above problem using inf-sup conforming finite elements. 
- E.g., Q2Q1 discretization where the velocity functions are biquadratic (9-noded quadrilateral) and pressure is approximated using bilinear (4-noded quadrilateral) elements.
- Parts (c) and (d) plots are included
"""
print(f"\nSolving part (b): Running simulation with Q2Q1 elements. \nVelocity functions (Q2) are biquadratic elements (9-noded quadrilateral) and pressure functions (Q1) are bilinear elements (4-noded quadrilateral).")

# Mesh
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

# Define function spaces for Q2Q1 
# Q2Q1: Quadratic elements for velocity and linear elements for pressure
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # Velocity space (vector field, Q2)
Q = FunctionSpace(mesh, "Lagrange", 1)        # Pressure space (scalar field, Q1)
element = MixedElement([    # Mixed element for velocity (vector field, Q2) and pressure (scalar field, Q1)
    VectorElement("Lagrange", mesh.ufl_cell(), 2),  # Q2: Quadratic elements for velocity
    FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Q1: Linear elements for pressure
])
W = FunctionSpace(mesh, element)  # Mixed function space

# Define trial and test functions for variational formulation
(u, p) = TrialFunctions(W)  # Trial functions for velocity (u) and pressure (p)
(v, q) = TestFunctions(W)  # Test functions for velocity (v) and pressure (q)

#print(f"Q2Q1 function spaces defined.\n")
print("\nVelocity function space (Q2) is a vector field (2D) with biquadratic elements (9-noded quadrilateral).")
print("Pressure function space is a scalar field (1D) with bilinear elements (4-noded quadrilateral).\n")

# Define the parabolic inflow velocity profile
U_max = 1
U_inlet = Expression(("U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=1.0, H=H, degree=2)  # U_max = centerline velocity
print(f"\nInflow velocity profile: {U_inlet}\n")

# Compute the velocity components across the domain
U_x = U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in x-direction
U_y = np.zeros_like(Y)                    # Zero velocity in the y-direction
print("\nAnalytical solution for inflow velocity:\n")
print(f"Parabolic inflow velocity in the x-direction:\n \n{U_x}\n")
print(f"Parabolic inflow velocity in the y-direction:\n \n{U_y}\n")

## Stokes Equations - weak/variational form

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

## Dirichlet Boundary Conditions
inflow_bc = DirichletBC(W.sub(0), U_inlet, "near(x[0], 0)")    # Inflow boundary: Parabolic velocity profile
walls_bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[1], 0 || near(x[1], {H}))".format(H=H))   # Walls (top and bottom): No-slip condition
outflow_bc = DirichletBC(W.sub(1), Constant(0.0), "near(x[0], {L})".format(L=L)) # Outflow boundary: Pressure 
bcs = [inflow_bc, walls_bc, outflow_bc] # Combine boundary conditions

print("\nDirichlet boundary conditions defined.")

# Q2Q1 Solver
print("\nSolving Q2Q1 solution...\n") 
w_Q2Q1 = Function(W) # Create a function to store the solution
solve(a_stokes == L_stokes, w_Q2Q1, bcs) # Solve the linear system 
(velocity_solution_Q2Q1, pressure_solution_Q2Q1) = w_Q2Q1.split()  # Extract velocity and pressure solutions

# Velocity solutions
velocity_Q2Q1 = velocity_solution_Q2Q1.compute_vertex_values(mesh)
#print(f"\nVelocity: \n{velocity_Q2Q1}")    # error handling
if not np.isfinite(velocity_Q2Q1).all():
    raise ValueError("\nVelocity contains non-finite values (NaN or Inf).\n")

# Pressure solutions 
pressure_Q2Q1 = pressure_solution_Q2Q1.compute_vertex_values(mesh)
#print(f"\nPressure: \n{pressure_Q2Q1}\n")    # error handling
if not np.isfinite(pressure_Q2Q1).all():
    raise ValueError("\nPressure contains non-finite values (NaN or Inf).\n")

# Compute the numerical pressure gradient along the centerline
#print("\nThe pressure gradient provides the slope of the pressure curve.")
x_coords_Q2Q1 = np.linspace(0, L, 100)  # Sample x-coordinates along the channel
pressure_gradient_Q2Q1 = (pressure_solution_Q2Q1(Point(x_coords[-1], H/2)) - pressure_solution_Q2Q1(Point(x_coords[0], H/2))) / L
print(f"Numerical Pressure Gradient: {pressure_gradient_Q2Q1:.4f}")

# Analytical pressure gradient - dp/dx = -2 * μ * U_max / (H**2)
print(f"Analytical Pressure Gradient = dp/dx = {analytical_pressure_gradient}") # solved in part a

# Relative error between numerical and analytical pressure gradients
relative_error_Q2Q1 = abs((pressure_gradient_Q2Q1 - analytical_pressure_gradient) / analytical_pressure_gradient) * 100
print(f"Relative Error: {relative_error_Q2Q1:.2f}%")

# Relative error between numerical and analytical velocity gradients
error = errornorm(U_inlet, velocity_solution_Q2Q1, norm_type="L2") * 100
print(F"Relative error for velocity: {error:.4f}%\n")

# Plot velocity field (magnitude)
print(f"\nVelocity solution plot: \n{velocity_solution_Q2Q1}")
plt.figure()
plot_u = plot(velocity_solution_Q2Q1*4, title="Velocity Field for Q2Q1", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_u, label="Velocity value")  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
# Save the plot 
filename = f"Velocity Field for Q2Q1.png"
save_path = os.path.join(save_folder, filename)
plt.savefig(save_path)
print(f"\nGraph saved as {save_path}.\n")
plt.show()

# Plot pressure field
print(f"\nPressure solution plot: \n{pressure_solution_Q2Q1}\n")
plt.figure()
plot_p = plot(pressure_solution_Q2Q1, title="Pressure Field for Q2Q1", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_p, label="Pressure value")  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
# Save the plot 
filename = f"Pressure Field for Q2Q1.png"
save_path = os.path.join(save_folder, filename)
plt.savefig(save_path)
print(f"\nGraph saved as {save_path}.\n")
plt.show()

## Plot the pressure along the centerline 
print(f"Pressure along the centerline plot:\n")
# Initialize arrays to store numerical and analytical pressure values
numerical_pressure_Q2Q1 = []
analytical_pressure_Q2Q1 = []

# Compute pressure values along the centerline
for x in x_coords:
    numerical_pressure_Q2Q1.append(pressure_solution_Q2Q1(Point(x, H/2)))  # Numerical pressure
    analytical_pressure_Q2Q1.append(p_o + analytical_pressure_gradient * x)  # Analytical pressure

# Plot the pressure distributions
plt.figure(figsize=(10, 6))
plt.plot(x_coords, numerical_pressure_Q2Q1, label="Numerical Pressure (P1P1)", color="blue", linewidth=2)
plt.plot(x_coords, analytical_pressure_Q2Q1, label="Analytical Pressure", color="red", linestyle="--", linewidth=2)
plt.xlabel("x (Length along channel)", fontsize=12)
plt.ylabel("Pressure (psi)", fontsize=12)
plt.title("Pressure Distribution Along the Centerline of the Channel for Q2Q1", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
# Save the plot 
filename = f"Pressure Distribution Along the Centerline of the Channel for Q2Q1.png"
save_path = os.path.join(save_folder, filename)
plt.savefig(save_path)
print(f"Graph saved as {save_path}.\n")
plt.show()