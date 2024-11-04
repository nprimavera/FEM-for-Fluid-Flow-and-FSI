#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, exp

# Solve the ADE using Galerkin FEM
class GFEM_AdvectionDiffusionSolver:

    """
    Nicolino Primavera 
    FEM for Fluid Flow and FSI Interactions
    Assignment 1 
    10/11/24 

    Using the Finite Element Method (FEM) to analyze the Steady-State Advection-Diffusion Equation
    Steady-state Advection-Diffusion Equation: (du/dx) = (0.05)*(d^2u/dx^2) - strong form, Domain: 0 <= x <= 1 , u(0) = 0 & u(1) = 10

    Want to explore the effect of cell Peclet number on the computed solution when Galerkin FEM is applied to this problem
    Pe number represents the ratio of advection to diffusion
    Use numerical inegration or Gauss quadrature-based integration (Gauss integration)

    Solve: 
        a. Choose at least four different grids for a range of cell Peclet numbers (Pe) and demonstrate that oscillatory solutions are obtained when Pe > 2
        b. Compare the solutions for Pe < 2 against the analytical solution 
    """

    print("\nStarting Program...\n")

    # Initialize parameters
    def __init__(self, Pe_values, a=1.0, k=0.05, x_0=0, x_1=1, u_0=0, u_1=10):
        self.Pe_values = Pe_values  # Cell Peclet Numbers
        self.a = a                  # Advection speed
        self.k = k                  # Diffusion coefficient 
        self.x_0 = x_0              # Domain
        self.x_1 = x_1              # Domain
        self.u_0 = u_0              # Boundary Condition
        self.u_1 = u_1              # Boundary Condition 
    
    # Gauss Quadrature (numerical integration)
    def gauss_quadrature(self, n_points):
        if n_points == 2:                                                       # 2-point Gauss Quadrature 
            return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1, 1])    # gauss quadrature points(2), weights 
        else:
            raise ValueError("Only 2-point quadrature implemented.")

    # Local Stiffness Matrix 
    def local_stiffness_matrix(self, h, Pe):
        K_local = np.zeros((2, 2))              # Initialize local stiffness matrix
        xi, w = self.gauss_quadrature(2)        # Gauss quadrature points (xi) , weights (w)
        # Nested for loops to compute and fill out the local stiffness matrices 
        for i in range(2):
            for j in range(2):
                diffusion = (1/h) * (1 if i == j else -1)                                       # Compute diffusion terms 
                advection = Pe * w[i] * (0.5 * (1 - xi[i]) if i == 0 else 0.5 * (1 + xi[i]))    # Compute advection terms 
                K_local[i, j] += diffusion + advection                                          # Append local stiffness matrices 
        return K_local

    # Global Stiffness Matrix 
    def assemble_global_matrices(self, N, h, Pe):
        K_global = np.zeros((N, N))                         # Initialize an empty array of size [N x N] (nodes) 
        for e in range(N-1):                                # Iterating through each element (1 element b/w every 2 nodes)
            K_local = self.local_stiffness_matrix(h, Pe)    # Assembling the local stiffness matrix for each element
            K_global[e:e+2, e:e+2] += K_local               # Appending each local stiffness matrix to the global stiffness matrix
        return K_global
    
    # Compute for each Cell Peclet Number 
    def solve(self):
        for Pe in self.Pe_values:
            h = (Pe * self.k) / self.a                               # Grid/Cell spacing - use uniform spacing
            N_no = int((self.x_1 - self.x_0) / h) + 1                # Computes number of nodes based on total domain length (x_1 - x_0)=1
            x = np.linspace(self.x_0, self.x_1, N_no)                # Computes nodal positions, N_no evenly spaced positions b/w the 1D domain boundaries
            K_global = self.assemble_global_matrices(N_no, h, Pe)    # Assembles the global stiffness matrix
            u = np.zeros(N_no)                                       # Initialize solution vector 'u' - stores the unknowns at each node
            u[0], u[-1] = self.u_0, self.u_1                         # Apply BCs
            K_global[0, :], K_global[-1, :] = 0, 0                   # Clear the first and last rows
            K_global[0, 0], K_global[-1, -1] = 1, 1                  # Set the first and last diagonal elements to 1
            u = np.linalg.solve(K_global, u)                         # Solves the system of linear eqns K_global(u)=g
            plt.plot(x, u, label=f'Pe = {Pe}')                       # Plots the Pe numbers
        
        # Solve the analytical solution 
        x_analytical = np.linspace(self.x_0, self.x_1, 100)
        u_analytical = 10 * x_analytical

        # Plot
        x_values, u_values = self.exact_solution_values()                               # Plot the exact solution
        plt.plot(x_analytical, u_analytical, 'k--', label='Analytical Solution')        # Plot the analytical solution
        plt.plot(x_values, u_values, label="Exact Solution", color="purple")            # Plot Peclet numbers 
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Galerkin FEM Solutions for the Steady-State Advection-Diffusion Equation')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Solve for the exact solution 
    def exact_solution_values(self):
        C1_value = 10 / (1 - np.exp(20))                    # C1 (constant) for exact solution
        x_values = np.linspace(0, 1, 100)                   # X values 
        u_values = C1_value * (1 - np.exp(20 * x_values))   # Exact solution in terms of x 
        return x_values, u_values

# Initialize and solve using Galerkin FEM
Pe_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7]   # Cell Peclet Numbers
solver = GFEM_AdvectionDiffusionSolver(Pe_values)       # Define solver 
solver.solve()                                          # Solve using solver function

"""
Nicolino Primavera 
FEM for Fluid Flow and FSI Interactions
Assignment 2
11/8/24 

Solve the same problem using the Exact Advection Diffusion (EAD) and streamwise upwind Petrov-Galerkin (SUPG) methods
    - use the same grids for the range of Pe numbers
    - compare the results against the exact solution
    - evaluate the effect of using higher-order (quadratic) elements
    - use element point of view to construct the stiffness matrix and load vector
    - write an assembly routine to assemble global stiffness matrix and load vector (already did previously)
    - use numerical integration at the element level (e.g. Gauss quadrature-based integration)
"""

# Solve the ADE using EAD method
class EAD_AdvectionDiffusionSolver:

    # Initialize parameters

# Solve the ADE using SUPG method 
class SUPG_AdvectionDiffusionSolver:

    # Initialize parameters 