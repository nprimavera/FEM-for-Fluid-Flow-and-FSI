#!/usr/bin/env python3

"""
Solve the unsteady diffusion problem in Fenics
    - use Galerkin FEM and write up the weak form
    - use any time integration scheme (explixit or implicit)
    - use any element type (linear, quadratic triangles, quadrilaterals with any order, etc.)

Environment:
    - square in shape that is traction free on the top and bottom surfaces
    - left side has u=10 and right side has u=0
    - domain is 100 x 100
    - u = 10, on x=0 and u=0 on x=100
    - PDE: du/dt = d^2u/dx^2 + d^2u/dy^2
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

from fenics import *
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.interpolate import interp1d 
import numpy as np
import matplotlib.pyplot as plt

# Parameters
domain = 100.0                      # Domain size
dt = 0.1                            # Time step size
T = 100.0                           # Final time
number_of_time_steps = int(T/dt)    # Time steps

# FE Mesh
#mesh = UnitIntervalMesh(50)                   # 1D mesh with 50 elements - (unit interval for a 1D problem or unit square for 2D)
mesh = RectangleMesh(Point(0, 0), Point(domain, domain), 50, 50)  # 2D mesh w/ 50 elements

# Function Space
V = FunctionSpace(mesh, 'P', 1)    # Linear elements
#V = FunctionSpace(mesh, 'P', 2)   # Quadratic elements

# Boundary Conditions (BCs)
u_left = Constant(10.0)           # Dirichlet BC on left side (u=10 & x=0)
u_right = Constant(0.0)           # Dirichlet BC on right side (u=0 & x=100)

# Define Left Boundary Function
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

# Define Right Boundary Function 
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], domain)

# Apply BCs
BC_left = DirichletBC(V, u_left, left_boundary)        # Left side of environment/square BC
BC_right = DirichletBC(V, u_right, right_boundary)     # Right side of environment/square BC
BCs = [BC_left, BC_right]                              # Boundary condition array

# Initial Condition
u_initial = interpolate(Constant(0.0), V)            # Initially u = 0 everywhere

# Weak Form - use the weak form of the unsteady diffusion equation with the implicit Euler method (define for each time step)
u = TrialFunction(V)  # Trial function - represents current solution at time step
v = TestFunction(V)   # Test function - used to construct the weak form of the PDE

# Weak form (LHS = RHS)
a = u * v * dx + dt * dot(grad(u), grad(v)) * dx    # Bilinear form 'a' (LHS)
L = u_initial * v * dx                              # Linear form 'L' (RHS)

# Function for the solution at each time step
u = Function(V)

# Prepare plots
fig, ax = plt.subplots()
plot_mesh = plot(u, mode="color", vmin=0, vmax=10)
plt.colorbar(plot(u, mode="color", vmin=0, vmax=10), ax=ax) 
#ax.set_title(f"Unsteady Diffusion Problem Solution")

# Simulation
def update_graph(frame):
    global u_initial, u     # Define global variables
    t = frame * dt          # graph x time step size

    solve(a == L, u, BCs)   # Solve the weak form 
    u_initial.assign(u)     # Update solution

    plot_mesh.set_array(u.compute_vertex_values(mesh))
    ax.set_title(f"Solution at t = {t:.2f}")
    return plot_mesh,

# Animation
ani = FuncAnimation(fig, update_graph, frames=number_of_time_steps, blit=True)    # Animation              
plt.xlabel("x")
plt.ylabel("y")

# Save animation
ani.save("diffusion_simulation.mp4", writer="ffmpeg", fps=60)
plt.show()