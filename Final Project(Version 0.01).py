#!/usr/bin/env python3

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

""" Problems:
(a) Simulate the above problem using stabilized finite element (FE) method using a reasonably chosen grid and equal order discretization for fluid velocity and pressure. E.g., P1P1 (linear triangles for both velocity and pressure), Q1Q1 (bilinear 4-noded quadrilaterals for both velocity and pressure, etc.)
(b) Simulate the above problem using inf-sup conforming finite elements. E.g., Q2Q1 discretization where the velocity functions are biquadratic (9-noded quadrilateral) and pressure is approximated using bilinear (4-noded quadrilateral) elements.
(c) Plot profiles of velocity and streamlines at any axial cross-section.
(d) Plot pressure along the centerline and compare the gradient against analytical expression assuming a fully developed parallel flow.
"""

# Initialize parameters
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)  # Rectangular mesh

# Define function spaces for velocity and pressure
# P1P1: Linear elements for both velocity and pressure
V = VectorFunctionSpace(mesh, "P", 1)  # Velocity space (vector field)
Q = FunctionSpace(mesh, "P", 1)        # Pressure space (scalar field)

# Define mixed element for velocity (vector field) and pressure (scalar field)
element = MixedElement([VectorElement("P", mesh.ufl_cell(), 1), FiniteElement("P", mesh.ufl_cell(), 1)])

# Create the mixed function space using the mixed element
W = FunctionSpace(mesh, element)

