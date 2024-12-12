#/Users/nicolinoprimavera

import numpy as np
import os
import matplotlib.pyplot as plt
from fenics import *
from dolfin import *

print("\nWorking directory:", os.getcwd())
print("\nStarting Program...\n")

"""
Nicolino Primavera
FEM for Fluid Flow and FSI Interactions
Final Project
12/17/24
"""

"""
Problem II: Stokes Flow in a 2D Channel

Objective: compare the numerically predicted pressure gradient against anyalytical value assuming stokes flow

Stokes flow governing equations:
    ρ(∂u/dt - f^b) = ∇ ∙ σ
    ∇ ∙ u = 0 (incompressibility)
    -∇p + μ∇^2u = 0     (∇^2u represents the Laplacian of velocity)

Stabilized Stokes Equations:
    - Petrov Galerkin method to enhance stability without losing consistency 
    - Galerkin formulation + add stabilization terms 
    - stabilization allows using equal-order discretization for both velocity and pressure function spaces (ex. P1-P1, Q1-Q1)

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
print(f"Channel dimensions: {L} cm x {H} cm")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)
print(f"\nFluid density: ρ = {ρ} g/cm^3 \nFluid viscosity: μ = {μ} g/cm/s")
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
t = 0.0              # initial time
T = 10               # Final time 
num_steps = 500      # number of time steps 
dt = T / num_steps   # time step size 

# Part (a)
# Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure.
# Ex: P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)
def P1P1_simulation():
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

    # Create trial and test functions - since we have two different function spaces, we need to create two sets of trial and test functions
    u = TrialFunction(V_P1)
    v = TestFunction(V_P1)
    p = TrialFunction(Q_P1)
    q = TestFunction(Q_P1)

    print("\nCreation of function spaces, trial and test functions complete.\n")

    # Define mixed element for velocity (vector field) and pressure (scalar field)
    element_1 = MixedElement([V_P1.ufl_element(), Q_P1.ufl_element()])  # uses equal-order interpolation (P1-P1) - linear triangles for both velocity and pressure
    W_P1 = FunctionSpace(mesh, element_1)   # Create the mixed function space using the mixed element for P1P1

    # Define the parabolic inflow velocity profile
    U_in = Expression(("4.0 * U_max * x[1] * (H - x[1]) / pow(H, 2)", "0.0"), U_max=1.0, H=H, degree=2)
    U_max = 1.0  # Maximum centerline velocity

    # Create a mesh grid for the whole domain
    x_vals = np.linspace(0, 8, 100) # L = 8
    y_vals = np.linspace(0, 1, 100) # H = 1
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute the velocity components across the domain
    U_x = np.zeros_like(X)                    # Zero velocity in the x-direction
    U_y = 4.0 * U_max * Y * (H - Y) / (H**2)  # Parabolic velocity profile in y-direction
    print(f"\nVelocity in the x-direction:\n \n{U_x}\n")
    print(f"Velocity in the y-direction:\n \n{U_y}\n")

    # Boundary conditions (BCs)
    # Boundary conditions: No-slip at walls, inflow at left, zero pressure at outflow
    def walls(x, on_boundary):
        return on_boundary and (near(x[1], 0) or near(x[1], H))
    bc_noslip = DirichletBC(V_P1, Constant((0, 0)), walls)  # No-slip condition for velocity

    def inflow(x, on_boundary):
        return on_boundary and near(x[0], 0)
    bc_inflow = DirichletBC(Q_P1, Constant(0), inflow)  # Pressure at inflow (assumed 0)

    def outflow(x, on_boundary):
        return on_boundary and near(x[0], L)
    bc_outflow = DirichletBC(Q_P1, Constant(0), outflow)  # Pressure at outflow

    bcs_P1 = [bc_noslip, bc_inflow, bc_outflow]

    print("\nApplication of boundary Conditions complete.\n")

    # Stokes Equations (weak/variational form) - for incompressible flow in a 2D domain  
    a_stokes = μ * inner(grad(u), grad(v)) * dx - div(v) * p * dx - div(u) * q * dx    # Bilinear form of Stokes equation
    L_stokes = dot(Constant((0.0, 0.0)), v) * dx                                       # Linear form (no body forces) of Stokes equation
    print(f"\nBilinear form of stokes equations:\n \n{a_stokes}") 
    print(f"\nLinear form of stokes equations:\n \n{L_stokes}\n")

    # Implement stabilization
    h = mesh.hmin()             # Minimum mesh size
    tau = Constant(5 * H/ μ)    # Stabilization parameter
    print(f"\nStabilization parameter: {tau}\n")

    # Stokes equations (stabilized)
    a_P1 = (                                  # bilinear form
        μ * inner(grad(u), grad(v)) * dx
        - div(v) * p * dx
        - div(u) * q * dx
        + tau * div(u) * div(v) * dx          # Stabilization for continuity
        + tau * p * q * dx                    # Pressure stabilization
    )
    L_P1 = dot(Constant((0.0, 0.0)), v) * dx  # linear form
    print(f"\nBilinear form of stokes equations with stabilization term: \n{a_P1}")
    print(f"\nLinear form of stokes equations with stabilization term: \n{L_P1}\n")

    # Compute analytical pressure gradient
    #dp/dx = -2 * μ * U_max / (H**2)
    analytical_pressure_gradient = -2 * μ * U_max / (H**2)
    print(f"\nAnalytical Pressure Gradient = dp/dx = {analytical_pressure_gradient} \nThis provides the slope of the pressure curve.\n")

    # Time-stepping loop
    u_new = Function(V_P1)
    p_new = Function(Q_P1)
    u_old = Function(V_P1)
    p_old = Function(Q_P1)
    u_old.assign(u_initial)
    p_old.assign(p_initial)
    u_new.assign(u_old)  # Initial guess for velocity (same as old)
    p_new.assign(p_old)  # Initial guess for pressure (same as old)

    # Time-stepping loop for transient problem
    print("\nBeginning simulation....\n")
    for n in range(num_steps):
        # Update the weak form for time-stepping (backward Euler for velocity)
        a_t = (1/dt * inner(u_new - u_old, v) + inner(grad(u_new), grad(v)) - p_new * div(v) - q * div(u_new)) * dx
        L_t = -inner(f_b, v) * dx  # Update for body force (none in this case)

        # Solve for the new velocity and pressure
        solve(a_t == L_t, [u_new, p_new], bcs=bcs_P1)

        # Update old solutions for the next time step
        u_old.assign(u_new)
        p_old.assign(p_new)

    print("\nSimulation complete.\n")


    # Post-processing: Plot velocity and pressure fields
    plot(u_new, title="Velocity Field")
    plt.show()
    plot(p_new, title="Pressure Field")
    plt.show()

    # Extracting velocity profiles for analysis 
    x_vals = np.linspace(0, L, 100)
    y_vals = np.linspace(0, H, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    velocity_profile = np.zeros_like(X)

    for i in range(len(x_vals)):
        velocity_profile[i] = u_new.compute_vertex_values(mesh)[:len(y_vals)]  # extract values at a specific x

    plt.plot(x_vals, velocity_profile)
    plt.title("Velocity Profile along the Centerline")
    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.show()

# Run the P1P1 simulation (part a)
P1P1_simulation()