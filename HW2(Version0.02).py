#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, exp

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
        self.Pe_values = Pe_values
        self.a = a
        self.k = k
        self.x_0 = x_0
        self.x_1 = x_1
        self.u_0 = u_0
        self.u_1 = u_1
    
    def gauss_quadrature(self, n_points):
        if n_points == 2:
            return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1, 1])
        else:
            raise ValueError("Only 2-point quadrature implemented.")

    def local_stiffness_matrix(self, h, Pe):
        K_local = np.zeros((2, 2))
        xi, w = self.gauss_quadrature(2)
        for i in range(2):
            for j in range(2):
                diffusion = (1/h) * (1 if i == j else -1)
                advection = Pe * w[i] * (0.5 * (1 - xi[i]) if i == 0 else 0.5 * (1 + xi[i]))
                K_local[i, j] += diffusion + advection
        return K_local

    def assemble_global_matrices(self, N, h, Pe):
        K_global = np.zeros((N, N))
        for e in range(N-1):
            K_local = self.local_stiffness_matrix(h, Pe)
            K_global[e:e+2, e:e+2] += K_local
        return K_global
    
    def solve(self):
        for Pe in self.Pe_values:
            h = (Pe * self.k) / self.a
            N_no = int((self.x_1 - self.x_0) / h) + 1
            x = np.linspace(self.x_0, self.x_1, N_no)
            K_global = self.assemble_global_matrices(N_no, h, Pe)
            u = np.zeros(N_no)
            u[0], u[-1] = self.u_0, self.u_1
            K_global[0, :], K_global[-1, :] = 0, 0
            K_global[0, 0], K_global[-1, -1] = 1, 1
            u = np.linalg.solve(K_global, u)
            plt.plot(x, u, label=f'Pe = {Pe}')
        
        x_analytical = np.linspace(self.x_0, self.x_1, 100)
        u_analytical = 10 * x_analytical
        x_values, u_values = self.exact_solution_values()
        plt.plot(x_analytical, u_analytical, 'k--', label='Analytical Solution')
        plt.plot(x_values, u_values, label="Exact Solution", color="purple")
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('FEM Solutions for the Steady-State Advection-Diffusion Equation')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def exact_solution_values(self):
        C1_value = 10 / (1 - np.exp(20))
        x_values = np.linspace(0, 1, 100)
        u_values = C1_value * (1 - np.exp(20 * x_values))
        return x_values, u_values

# Initialize and solve using Galerkin FEM
Pe_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7]
solver = GFEM_AdvectionDiffusionSolver(Pe_values)
solver.solve()

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

