
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import dolfin

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from fenics import *

## Stokes flow governing equations
#ρ * (du/dt - f_b) = ∇ * 𝝈
#∇ * u = 0

# Parameters
#𝝈_ij = -p * kronecker_delta + μ * (u_ij + u_ji) # Cauchy stress for the fluid
#u = fluid velocity
#p = fluid pressure
#f_b = body force acting on the fluid

# Initialize channel parameters
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
print(f"Channel dimensions: {L} cm x {H} cm")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)
print(f"\nFluid density: ρ = {ρ} g/cm^3 \nFluid viscosity: μ = {μ} g/cm/s")
U_infinity = 1  # fluid velocity (cm/s) at the centerline - inflow

# Incompressible flow
f_b = 0   # no body force
print(f"\nIncompressible flow = no body force (f_b = {f_b} N)")

# Reynolds number
Re = ρ * U_infinity * H / μ
print(f"\nReynolds number: Re = {Re} and therefore, a Stokes flow is a reasonable assumption")

# Pressure
p_o = 0   # initial pressure
p_f = -16 # final pressure (analytical solution)
print(f"\nAnalytical Solution:\n  If the initial pressure: p_o = {p_o}psi, then the final pressure: p_f = {p_f}psi")
print(f"p = p_o - 2 * L\n")

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)  # 2D mesh

# For a 2D mesh, retrieve the number of elements in each direction (nx, ny)
coords = mesh.coordinates()
x_coords = coords[:, 0]  # x-coordinates of all points
y_coords = coords[:, 1]  # y-coordinates of all points

# Number of unique points in each direction
unique_x = np.unique(x_coords)
unique_y = np.unique(y_coords)

print(f"Number of elements along x: {len(unique_x) - 1}")
print(f"Number of elements along y: {len(unique_y) - 1}\n")

# Part (a)
# Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure.
# Ex: P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)

# P1P1 Simulation with Stabilization
print("\nSolving part (a): Running simulation with P1P1 linear elements (equal-order)...\n")

# Define function spaces for velocity and pressure (P1P1)
V_P1 = VectorFunctionSpace(mesh, "P", 1)  # Velocity space (vector field, linear elements)
Q_P1 = FunctionSpace(mesh, "P", 1)       # Pressure space (scalar field, linear elements)

# Define initial condition for simulation
u_initial = Function(V_P1)                # previous time step for velocity field (vector)
u_initial.assign(Constant((0.0, 0.0)))    # initial state (vector field)
p_initial = Function(Q_P1)                # previous time step for pressure field (scalar)
p_initial.assign(Constant(0.0))           # initial state (scalar field)
# Example pressure solution
#p_o = 0   # initial pressure
#for i in
  #p = p_o - 2 * L[i]
  #i+1

# Define mixed element for velocity (vector field) and pressure (scalar field)
element_1 = MixedElement([V_P1.ufl_element(), Q_P1.ufl_element()])  # uses equal-order interpolation (P1-P1) - linear triangles for both velocity and pressure

# Create the mixed function space using the mixed element for P1P1
W_P1 = FunctionSpace(mesh, element_1)

# Define the parabolic inflow velocity profile
U_in = Expression(("4.0 * U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=1.0, H=H, degree=2)
U_max = 1.0  # Maximum centerline velocity

# Create a mesh grid for the whole domain
x_vals = np.linspace(0, L, 100)
y_vals = np.linspace(0, H, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the velocity components across the domain
U_x = np.zeros_like(X)                    # Zero velocity in the x-direction
U_y = 4.0 * U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in y-direction

# Boundary conditions (BCs)
bc_inflow = DirichletBC(W_P1.sub(0), U_in, "near(x[0], 0)"),                          # Inflow
bc_walls = DirichletBC(W_P1.sub(0), Constant((0.0, 0.0)),                             # No-slip walls
                "on_boundary && (near(x[1], 0) || near(x[1], {H}))".format(H=H)),
bc_pressure = DirichletBC(W_P1.sub(1), Constant(0.0), "near(x[0], 0)")                # Pressure reference

bcs_P1 = [bc_inflow, bc_walls, bc_pressure]

# Split into trial and test functions
(u, p) = TrialFunctions(W_P1) # Trial function - represents current solution at time step
(v, q) = TestFunctions(W_P1)  # Test function - used to construct the weak form of the PDE

# Stokes Equations
a_stokes = μ * inner(grad(u), grad(v)) * dx - div(v) * p * dx - div(u) * q * dx    # Bilinear form of Stokes equation
L_stokes = dot(Constant((0.0, 0.0)), v) * dx                                       # Linear form (no body forces) of Stokes equation

# Implement stabilization
h = mesh.hmin()             # Minimum mesh size
tau = Constant(5 * h/ μ)    # Stabilization parameter

# Stokes equations
a_P1 = (                                  # bilinear form
    μ * inner(grad(u), grad(v)) * dx
    - div(v) * p * dx
    - div(u) * q * dx
    + tau * div(u) * div(v) * dx          # Stabilization for continuity
    + tau * p * q * dx                    # Pressure stabilization
)

L_P1 = dot(Constant((0.0, 0.0)), v) * dx  # linear form

# Compute analytical pressure gradient
#dp/dx = -2 * μ * U_max / (H**2)
analytical_pressure_gradient = -2 * μ * U_max / (H**2)
print(f"\nAnalytical Pressure Gradient = dp/dx = {analytical_pressure_gradient} \nThis provides the slope of the pressure curve")

# Solve the P1P1 system
w_P1 = Function(W_P1)               # create a function to store the solution

# Extract velocity and pressure solutions
(u_P1, p_P1) = w_P1.split()

# Initialize pressure plot for simulation
fig, ax = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
p_array = p_initial.compute_vertex_values(mesh).reshape(len(unique_x), len(unique_y))
img = ax.imshow(p_array, extent=[0, L, 0, H], origin="lower", cmap="viridis")
plt.colorbar(img, ax=ax)

# Parameters
T = 200.0                           # Final time
t = 0.0                             # Initialize time
dt = 0.1                            # Time step size
number_of_time_steps = int(T/dt)    # Time steps

# Update pressure function for animation
def update(n):
    global t
    t += dt
    solve(a_P1 == L_P1, w_P1, bcs_P1)
    p_initial.assign(p_P1)

    p_array = p_P1.compute_vertex_values(mesh).reshape(len(unique_x), len(unique_y))
    img.set_data(p_array)
    ax.set_title(f"Pressure Field (scalar field) at time: {t:.2f}s")

# Create the animation
ani = FuncAnimation(fig, update, frames=number_of_time_steps, interval=100)

# Save animation
ani.save('velocity_field.mp4', writer='ffmpeg', fps=10)

# Initialize velocity plot for simulation
fig, ax = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
u_array = u_initial.compute_vertex_values(mesh).reshape(len(unique_x), len(unique_y))
img = ax.imshow(u_array, extent=[0, L, 0, H], origin="lower", vmin=0, vmax=1, cmap="viridis")
plt.colorbar(img, ax=ax)

# Update velocity function for animation
def update(n):
    global t
    t += dt
    solve(a_P1 == L_P1, w_P1, bcs_P1)
    u_initial.assign(u_P1)

    u_array = u_P1.compute_vertex_values(mesh).reshape(len(unique_x), len(unique_y))
    img.set_data(u_array)
    ax.set_title(f"Velocity Field at Time: {t:.2f}s")

# Create the animation
ani = FuncAnimation(fig, update, frames=number_of_time_steps, interval=100)

# Save animation
ani.save('velocity_field.mp4', writer='ffmpeg', fps=10)



# Plot pressure field
print(f"Pressure solution: \n{p_P1}")
plt.figure()
plot_p = plot(p_P1, title="Pressure Field (P1P1)", cmap=cm.viridis)  # Directly plot the pressure field
plt.colorbar(plot_p)  # Attach the colorbar to the mappable object
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot velocity field (magnitude)
print(f"\nVelocity solution: \n{u_P1}")
plt.figure()
u_magnitude = sqrt(inner(u_P1, u_P1))  # Correct velocity magnitude calculation
plot_u = plot(u_magnitude, title="Velocity Magnitude", cmap=cm.viridis)
plt.colorbar(plot_u)
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

