# -*- coding: utf-8 -*-
"""FEM Final Project (Version 0.01).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zecje0hvk3PPB6fG-I25GIIcakODGoAO
"""

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

# Execute function
fenics()

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import pyplot as plt
from matplotlib import cm
from fenics import *

"""
Nicolino Primavera
FEM for Fluid Flow and FSI Interactions
Final Project
12/17/24
"""

print("\nStarting Program...\n")

"""
Problem II: Stokes Flow in a 2D Channel

Objective: compare the numerically predicted pressure gradient against anyalytical value assuming stokes flow

Stokes flow governing equations:
    ρ(∂u/dt - f^b) = ∇ ∙ σ
    ∇ ∙ u = 0 (incompressibility)
    -∇p + μ∇^2u = 0     (∇^2u represents the Laplacian of velocity)

Variables:
    σ_ij = -p*δ_ij + μ(u_i,j + u_j,i) - Cauchy stress for the fluid
    u - fluid velocity
    p - fluid pressure
    f^b - body force acting on the fluid
    ρ - fluid density
    μ - dynamic viscosity

Channel dimensions: 1cm (height) x 8cm (length)

Boundary Conditions:
    Inflow: Dirichlet condition with a velocity profile u/U_∞ = y/H * (1 - y/H) where U_∞ = 1 cm/s is the velocity at the centerline
    Outflow: Traction-free boundary condition
    Top/Bottom faces: Fixed walls (no slip / no penetration condition, u = 0)

Physical Parameters:
    Incompressible flow, no body force (f^b = 0)
    Fluid Density (ρ): 1.0 g/cm^3
    Fluid Viscosity (μ): 1.0 g/cm/s

The Reynolds number of the flow Re = (ρ)*(U_∞)*(H)/μ = 1, therefore, a Stokes flow is a reasonable assumption for the above configuration
"""

"""Problems:
- (a) Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure. E.g., P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)

- (b) Simulate the above problem using inf-sup conforming finite elements. E.g., Q2Q1 discretization where the velocity functions are biquadratic (9-noded quadrilateral) and pressure is approximated using bilinear (4-noded quadrilateral) elements.
- (c) Plot profiles of velocity and streamlines at any axial cross-section.
- (d) Plot pressure along the centerline and compare the gradient against analytical expression assuming a fully developed parallel flow.
"""

# Initialize channel parameters
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
print(f"Channel dimensions: {L} cm x {H} cm\n")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)
print(f"Fluid parameters: ρ = {ρ} g/cm^3, μ = {μ} g/cm/s")

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

# General information about the mesh
print("Mesh Information:")
print(mesh)

# Print the number of cells (elements) and vertices (nodes)
print(f"\nNumber of cells (elements): {mesh.num_cells()}")
print(f"Number of vertices (nodes): {mesh.num_vertices()}\n")

# Print the geometric dimensions of the mesh
print(f"Mesh geometry dimension: {mesh.geometry().dim()}\n")
print(f"Mesh coordinates:\n {mesh.coordinates()}\n")
print(f"Mesh cell vertices:\n {mesh.cells()}\n")

# For a 2D mesh, retrieve the number of elements in each direction (nx, ny)
coords = mesh.coordinates()
x_coords = coords[:, 0]  # x-coordinates of all points
y_coords = coords[:, 1]  # y-coordinates of all points

# Number of unique points in each direction
unique_x = np.unique(x_coords)
unique_y = np.unique(y_coords)

print(f"Number of elements along x: {len(unique_x) - 1}")
print(f"Number of elements along y: {len(unique_y) - 1}\n")

# Plot
plt.plot(x_coords, y_coords, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mesh Points')
plt.grid(True)
plt.show()

# Define function spaces for velocity and pressure

# P1P1: Linear elements for both velocity and pressure
V = VectorFunctionSpace(mesh, "P", 1)  # Velocity space (vector field)
Q = FunctionSpace(mesh, "P", 1)        # Pressure space (scalar field)

# Error handling
#print(f"Velocity space: \n{V}\n")
#print(f"Pressure space: \n{Q}\n")

# Plots

# Velocity space (represents a vector field)
u_zero = Function(V)    # Create a zero vector function in the velocity space
plot(u_zero, title="Velocity Space Basis Functions (P1)")   # Plotting a zero field gives an idea of the mesh layout
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Pressure space (represents a scalar field)
p_zero = Function(Q)  # Create a zero scalar function in the pressure space
plot(p_zero, title="Pressure Space Basis Functions (P1)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot the underlying mesh structure used in these function spaces
plot(mesh, title="Mesh Structure")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Label nodes and elements
coords = mesh.coordinates()
for i, (x, y) in enumerate(coords):
    plt.text(x, y, str(i), color="red")  # Label nodes with their indices
    plt.title("Mesh Structure with Node Labels")

plt.show()

# Define mixed element for velocity (vector field) and pressure (scalar field)
element = MixedElement([VectorElement("P", mesh.ufl_cell(), 1), FiniteElement("P", mesh.ufl_cell(), 1)])

print(f"Mixed element: \n{element}\n")

# Create the mixed function space using the mixed element
W = FunctionSpace(mesh, element)

print(f"Mixed function space: \n{W}\n")

# Plot the fields

# Create a zero function in the mixed space
w_test = Function(W)

# Split into velocity and pressure components
u_mixed, p_mixed = w_test.split()

# Plot the velocity field (vector field)
velocity_plot = plot(u_mixed, title="Velocity Field", cmap='viridis')  # You can choose the colormap
plt.colorbar(velocity_plot)  # Attach the colorbar to the plot
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot the pressure field (scalar field)
pressure_plot = plot(p_mixed, title="Pressure Field", cmap='viridis')  # You can choose the colormap
plt.colorbar(pressure_plot)  # Attach the colorbar to the plot
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Define the parabolic inflow velocity profile
U_in = Expression(("4.0 * U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=1.0, H=H, degree=2)  # U_max = centerline velocity

print(f"Inflow velocity profile: \n{U_in}\n")

# Create a mesh grid for the whole domain
x_vals = np.linspace(0, L, 100)
y_vals = np.linspace(0, H, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the velocity components across the domain
U_x = np.zeros_like(X)  # Zero velocity in the x-direction
U_y = 4.0 * U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in y-direction

# Plot the velocity field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U_x, U_y, scale=5, color='blue')
plt.title("Inflow Velocity Profile (Vector Field)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Apply boundary conditions

# Inflow boundary: Parabolic velocity profile
bcu_inflow = DirichletBC(W.sub(0), U_in, "near(x[0], 0)")

# Walls (top and bottom): No-slip condition
bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)),
                        "on_boundary && (near(x[1], 0) || near(x[1], {H}))".format(H=H))

# Combine boundary conditions
bcs = [bcu_inflow, bcu_walls]

# Error handling
#print(f"Inflow boundary condition: \n{bcu_inflow}\n")
#print(f"Walls boundary condition: \n{bcu_walls}\n")
#print(f"Combined boundary conditions: \n{bcs}\n")

# Define trial and test functions for velocity and pressure
(u, p) = TrialFunctions(W)  # Trial functions (solution)
(v, q) = TestFunctions(W)   # Test functions

# Error handling
#print(f"Trial functions (u, p): \n{u, p}\n")
#print(f"Test functions (v, q): \n{v, q}\n")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)

# Bilinear form
a = μ * inner(grad(u), grad(v)) * dx - div(v) * p * dx - div(u) * q * dx

# Linear form (no body forces)
L = dot(Constant((0.0, 0.0)), v) * dx

# Error handling
#print(f"Bilinear form: \n{a}\n")
#print(f"Linear form: \n{L}\n")

# Create a function to store the solution
w = Function(W)

print(f"Solution function: \n{w}\n")

# Solve the linear system
solve(a == L, w, bcs)

# Extract velocity and pressure solutions
(u_sol, p_sol) = w.split()

print(f"Velocity solution: \n{u_sol}\n")
print(f"Pressure solution: \n{p_sol}\n")

# Plot velocity field (magnitude)
print(f"Velocity solution: \n{u_sol}\n")

plt.figure()
u_magnitude = sqrt(dot(u_sol, u_sol))
plot_u = plot(u_magnitude, title="Velocity Field", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_u)  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")

# Plot pressure field
print(f"Pressure solution: \n{p_sol}\n")

plt.figure()
p_magnitude = sqrt(dot(p_sol, p_sol))
plot_p = plot(p_magnitude, title="Pressure Field", cmap=cm.viridis)  # Use a colormap
plt.colorbar(plot_p)  # Attach the colorbar to the mappable object
plot(p_sol, title="Pressure Field")
plt.xlabel("x")
plt.ylabel("y")

# Streamlines of the velocity field
plt.figure()
streamlines = plot(u_sol, title="Streamlines")
plt.colorbar(streamlines)
plt.show()

# Compute the numerical pressure gradient along the centerline
x_coords = np.linspace(0, L, 100)  # Sample x-coordinates along the channel
pressure_gradient = (p_sol(Point(x_coords[-1], H/2)) - p_sol(Point(x_coords[0], H/2))) / L
print(f"Numerical Pressure Gradient: {pressure_gradient}")

# === Step 7: Validation ===
# Compare numerical pressure gradient with analytical value
# Analytical pressure gradient for fully developed flow
analytical_pressure_gradient = -2 * mu * 1.0 / H**2  # U_max = 1 cm/s
print(f"Analytical Pressure Gradient: {analytical_pressure_gradient}")

# Relative error between numerical and analytical pressure gradients
relative_error = abs((pressure_gradient - analytical_pressure_gradient) / analytical_pressure_gradient) * 100
print(f"Relative Error: {relative_error:.2f}%")