#!/usr/bin/env python3

"""
Solve the unsteady diffusion problem in Fenics
    - use Galerkin FEM and write up the weak form
    - use any time integration scheme (explixit or implicit)
    - use any element type (linear, quadratic triangles, quadrilaterals with any order, etc.)

Environment:
    - square in shape that is traction free on the top and bottom surfaces
    - Domain: Ω=[0,100]×[0,100]
    - Left side: u = 10, on x=0
    - Right side: u=0 on x=100
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
domain_length = 100   # Domain size
dt = 0.1              # Time step

# Define mesh and function space
mesh = RectangleMesh(Point(0, 0), Point(domain_length, domain_length), 50, 50)  # 2D mesh w/ 50 elements
V = FunctionSpace(mesh, 'P', 1)                                                 # Linear
#V = FunctionSpace(mesh, 'P', 2)                                                # Quadratic elements

# Boundary Conditions (BCs)
u_left = Constant(10.0)         # Dirichlet BC on left side (u=10 & x=0)
u_right = Constant(0.0)         # Dirichlet BC on right side (u=0 & x=100)

# Define Left Boundary Function
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0)

# Define Right Boundary Function
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], domain_length)

# Apply BCs
bc_left = DirichletBC(V, u_left, left_boundary)       # Left side of environment/square BC
bc_right = DirichletBC(V, u_right, right_boundary)    # Right side of environment/square BC
bcs = [bc_left, bc_right]                             # Boundary condition array

# Initial condition
u_initial = Function(V)                     # Previous time step
u_initial.assign(Constant(0.0))             # Initial state
#u_initial = interpolate(Constant(0.0), V)   # Initially u = 0 everywhere

# Define variational/weak form problem - use the weak form of the unsteady diffusion equation with the implicit Euler method (define for each time step)
u = TrialFunction(V)  # Trial function - represents current solution at time step
v = TestFunction(V)   # Test function - used to construct the weak form of the PDE

# Variational form for time-stepping
a = u * v * dx + dt * dot(grad(u), grad(v)) * dx    # Bilinear form 'a' (LHS)
L = u_initial * v * dx                              # Linear form 'L' (RHS)

# Function for the solution at each time step
u = Function(V)

# Initialize plot for simulation 
fig, ax = plt.subplots()
plt.xlabel("x")
plt.ylabel("y")
#ax.set_title(f"Unsteady Diffusion Problem Solution")
#u_array = u_initial.compute_vertex_values(mesh).reshape((51, 51))  # Converts FEniCS function to numpy array for plotting
img = ax.imshow(u_array, extent=[0, domain_length, 0, domain_length], origin="lower", vmin=0, vmax=10, cmap="viridis")
plt.colorbar(img, ax=ax)

# Parameters
T = 200.0                           # Final time
t = 0.0                             # Initialize time
number_of_time_steps = int(T/dt)    # Time steps

# Update function for animation
def update(n):
    global t
    t += dt
    solve(a == L, u, bcs)
    u_initial.assign(u)
    
    # Update the image with new data
    u_array = u.compute_vertex_values(mesh).reshape((51, 51))
    img.set_data(u_array)
    ax.set_title(f"Unsteady Diffusion Solution at t = {T:.2f}")

# Create the animation
ani = FuncAnimation(fig, update, frames=number_of_time_steps, interval=100)

# Save the animation
ani.save('unsteady_diffusion_solution.mp4', writer='ffmpeg', fps=10)

plt.show()