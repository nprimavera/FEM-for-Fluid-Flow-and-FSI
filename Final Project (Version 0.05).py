# -*- coding: utf-8 -*-
"""Final project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wjpWszog4Ix8Wj7UFzkhlD7V6mLIJ_Vx
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
fenics()    # Execute function

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import pyplot as plt
from matplotlib import cm
from fenics import *

# Initialize channel parameters
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
print(f"Channel dimensions: {L} cm x {H} cm")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)
print(f"\nFluid parameters: ρ = {ρ} g/cm^3, μ = {μ} g/cm/s")

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

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

# Print mesh information
#print("\nMesh Information:")
#print(mesh)

# Print the number of cells (elements) and vertices (nodes)
#print(f"\nNumber of cells (elements): {mesh.num_cells()}")
#print(f"Number of vertices (nodes): {mesh.num_vertices()}\n")

# Print the geometric dimensions of the mesh
#print(f"Mesh geometry dimension: {mesh.geometry().dim()}")
#print(f"Mesh coordinates:\n {mesh.coordinates()}")
#print(f"Mesh cell vertices:\n {mesh.cells()}\n")

# Part (a)
# Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure.
# Ex: P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)


# P1P1 Simulation with Stabilization
def P1P1_simulation_with_stabilization():
    # Starting simulation
    print("\nSolving part (a): Running simulation with P1P1 linear elements (equal-order)...\n")

    # Define function spaces for velocity and pressure (P1P1)
    V_P1 = VectorFunctionSpace(mesh, "P", 1)  # Velocity space (vector field, linear elements)
    Q_P1 = FunctionSpace(mesh, "P", 1)       # Pressure space (scalar field, linear elements)
    #print(f"Velocity space (vector field, linear elements):\n   {V_P1}")
    #print(f"Pressure space (scalar field, linear elements):\n   {Q_P1}\n")

    # Plot the velocity and pressure function spaces
    def initial_function_space_plots():
        # Velocity space (represents a vector field)
        u_zero = Function(V_P1)    # Create a zero vector function in the velocity space
        plot(u_zero, title="Velocity Space Basis Functions (P1)")   # Plotting a zero field gives an idea of the mesh layout
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # Pressure space (represents a scalar field)
        p_zero = Function(Q_P1)  # Create a zero scalar function in the pressure space
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
    #initial_function_space_plots()

    # Define mixed element for velocity (vector field) and pressure (scalar field)
    element_1 = MixedElement([V_P1.ufl_element(), Q_P1.ufl_element()])  # uses equal-order interpolation (P1-P1) - linear triangles for both velocity and pressure
    #print(f"\nMixed element: \n {element_1}\n")

    # Create the mixed function space using the mixed element for P1P1
    W_P1 = FunctionSpace(mesh, element_1)
    #print(f"Mixed function space: \n  {W_P1}\n")

    # Create a zero function in the mixed space
    w_test_1 = Function(W_P1)
    # Split into velocity and pressure components
    u_mixed, p_mixed = w_test_1.split()

    # Plot the velocity (vector) and pressure (scalar) fields
    def velocity_pressure_fields():
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
    #velocity_pressure_fields()

    # Define the parabolic inflow velocity profile
    U_in = Expression(("4.0 * U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=1.0, H=H, degree=2)  # U_max = centerline velocity
    U_max = 1.0  # Maximum centerline velocity
    #print(f"Inflow velocity profile: \n{U_in}\n")

    # Create a mesh grid for the whole domain
    x_vals = np.linspace(0, L, 100)
    y_vals = np.linspace(0, H, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute the velocity components across the domain
    U_x = np.zeros_like(X)  # Zero velocity in the x-direction
    U_y = 4.0 * U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in y-direction

    # Plot the velocity field
    #plt.figure(figsize=(8, 6))
    #plt.quiver(X, Y, U_x, U_y, scale=5, color='blue')
    #plt.title("Inflow Velocity Profile (Vector Field)")
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.grid(True)
    #plt.show()

    # Boundary conditions
    bcs_P1 = [
    DirichletBC(W_P1.sub(0), U_in, "near(x[0], 0)"),                                # Inflow
    DirichletBC(W_P1.sub(0), Constant((0.0, 0.0)),                                  # No-slip walls
                "on_boundary && (near(x[1], 0) || near(x[1], {H}))".format(H=H)),
    DirichletBC(W_P1.sub(1), Constant(0.0), "near(x[0], 0)")                        # Pressure reference
    ]

    # Split into trial and test functions
    (u, p) = TrialFunctions(W_P1)
    (v, q) = TestFunctions(W_P1)
    #print(f"Trial functions (u, p): \n  {u}, {p}\n")
    #print(f"Test functions (v, q): \n  {v}, {q}\n")

    # Stokes Equations
    a_stokes = μ * inner(grad(u), grad(v)) * dx - div(v) * p * dx - div(u) * q * dx    # Bilinear form of Stokes equation
    L_stokes = dot(Constant((0.0, 0.0)), v) * dx       # Linear form (no body forces) of Stokes equation
    #print(f"Bilinear form: \n{a}\n")
    #print(f"Linear form: \n{L}\n")

    # Implement stabilization
    h = mesh.hmin()  # Minimum mesh size
    tau = Constant(h * 5 / μ)  # Stabilization parameter

    a_P1 = (                            # bilinear form
        μ * inner(grad(u), grad(v)) * dx
        - div(v) * p * dx
        - div(u) * q * dx
        + tau * div(u) * div(v) * dx    # Stabilization for continuity
        + tau * p * q * dx              # Pressure stabilization
    )

    L_P1 = dot(Constant((0.0, 0.0)), v) * dx    # linear form

    # Solve the P1P1 system
    w_P1 = Function(W_P1)               # create a function to store the solution

    solve(a_P1 == L_P1, w_P1, bcs_P1)   # solve with stabilization
    #solve(a_stokes == L_stokes, w_P1, bcs_P1)   # solve using stokes equations

    # Extract velocity and pressure solutions
    (u_P1, p_P1) = w_P1.split()
    #print(f"Velocity solution: \n{u_P1}\n")
    #print(f"Pressure solution: \n{p_P1}\n")

    # Compute analytical pressure gradient
    U_max = 1.0  # Maximum velocity at centerline (cm/s)
    analytical_pressure_gradient = -2 * μ * U_max / (H**2)
    print(f"\nAnalytical Pressure Gradient: {analytical_pressure_gradient}")

    # Plot velocity field (magnitude)
    print(f"\nVelocity solution: \n{u_P1}")
    plt.figure()
    u_magnitude = sqrt(inner(u_P1, u_P1))  # Correct velocity magnitude calculation
    plot_u = plot(u_magnitude, title="Velocity Magnitude", cmap=cm.viridis)
    plt.colorbar(plot_u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Plot pressure field
    print(f"Pressure solution: \n{p_P1}")
    plt.figure()
    plot_p = plot(p_P1, title="Pressure Field (P1P1)", cmap=cm.viridis)  # Directly plot the pressure field
    plt.colorbar(plot_p)  # Attach the colorbar to the mappable object
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Compute the numerical pressure gradient
    pressure_at_start = p_P1(Point(float(0), float(H / 2)))  # Start of the centerline
    pressure_at_end = p_P1(Point(float(L), float(H / 2)))   # End of the centerline
    print(f"\nPressure at start (x=0): {pressure_at_start}")
    print(f"\nPressure at end (x=L): {pressure_at_end}")
    print(f"\nChannel Length (L): {L}\n")

    # Compute pressure gradient
    pressure_gradient_P1 = (pressure_at_end - pressure_at_start) / L
    print(f"Numerical pressure gradient = (pressure at the end - pressure at the start) / L")
    print(f"Numerical Pressure Gradient (P1P1): {pressure_gradient_P1}")

    # Compute relative error compared to the analytical gradient
    relative_error_P1 = abs(
        (pressure_gradient_P1 - analytical_pressure_gradient) / analytical_pressure_gradient
    ) * 100
    print(f"\nAnalytical Pressure Gradient: {analytical_pressure_gradient}")
    print(f"Relative Error = (pressure gradient - analytical pressure gradient solution) / analytical pressure gradient solution")
    print(f"Relative Error (P1P1): {relative_error_P1:.2f}%\n")

    # Check pressure solutions
    p_values = p_P1.vector().get_local()
    print("Min pressure:", np.min(p_values))
    print("Max pressure:", np.max(p_values))
    print("Any NaNs:", np.any(np.isnan(p_values)))
    print("Any Infs:", np.any(np.isinf(p_values)))

P1P1_simulation_with_stabilization()

# Part (b)
# Simulate the above problem using inf-sup conforming finite elements.
# Ex: Q2Q1 discretization where the velocity functions are biquadratic (9-noded quadrilateral) and pressure is approximated using bilinear (4-noded quadrilateral) elements.

"""
PSPG - Pressure velocity decoupling problem --> "inf-sup" (Stokes equation)
"""

# Part (c)
# Plot profiles of velocity and streamlines at any axial cross-section.

def velocity_profiles_and_streamlines():
    # Streamlines of the velocity field
    plt.figure()
    streamlines = plot(u_solution, title="Streamlines")
    plt.colorbar(streamlines)
    plt.show()
#velocity_profiles_and_streamlines()

# Part (d)
# Plot pressure along the centerline and compare the gradient against analytical expression assuming a fully developed parallel flow.

def part_d():
  # Compute the numerical pressure gradient along the centerline
  x_coords = np.linspace(0, L, 100)  # Sample x-coordinates along the channel
  pressure_gradient = (p_solution(Point(x_coords[-1], H/2)) - p_solution(Point(x_coords[0], H/2))) / L
  print(f"Numerical Pressure Gradient: {pressure_gradient}")

  # analytical solution
  analytical_pressure_gradient = -2 * mu * 1.0 / H**2  # Analytical pressure gradient for fully developed flow
  print(f"Analytical Pressure Gradient: {analytical_pressure_gradient}")

  # Relative error between numerical and analytical pressure gradients
  relative_error = abs((pressure_gradient - analytical_pressure_gradient) / analytical_pressure_gradient) * 100
  print(f"\nRelative Error: {relative_error:.2f}%\n")

