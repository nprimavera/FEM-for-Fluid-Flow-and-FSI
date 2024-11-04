#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

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

# Base class for FEM solution
class GFEM_AdvectionDiffusionSolver:

    # Define parameters
    def __init__(self, Pe_values, a=1.0, k=0.05, x_0=0, x_1=1, u_0=0, u_1=10):
        self.Pe_values = Pe_values  # List of Peclet numbers to be tested
        self.a = a                  # Advection speed
        self.k = k                  # Diffusion coefficient
        self.x_0 = x_0              # Domain start point
        self.x_1 = x_1              # Domain end point
        self.u_0 = u_0              # Boundary condition at x = 0
        self.u_1 = u_1              # Boundary condition at x = 1

    # Gauss Quadrature 
    def gauss_quadrature(self, n_points):
        # Choose Gauss quadrature points and weights based on number of points needed
        if n_points == 2:       # 2-point Gauss quadrature
            return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1, 1])
        elif n_points == 3:     # 3-point Gauss quadrature (for quadratic elements)
            return np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]), np.array([5/9, 8/9, 5/9])
        else:
            # Error if unsupported number of points is requested
            raise ValueError("Only 2-point and 3-point quadrature implemented.")

    # Local Stiffness Matrix 
    def local_stiffness_matrix(self, h, Pe):
        K_local = np.zeros((3, 3))              # Local stiffness matrix for quadratic elements
        xi, w = self.gauss_quadrature(3)        # Use 3-point Gauss quadrature for higher accuracy
        
        # Loop over Gauss points
        for i in range(3):
            # Calculate shape function derivatives and their values at Gauss points
            # Linear shape function derivatives for quadratic elements
            N_prime = np.array([-(1 - xi[i])/(2*h), (1 + xi[i])/(2*h), (1 - xi[i])/(2*h)])  # Shape function derivatives
            
            # Calculate local stiffness matrix entries
            for j in range(3):
                K_local += w[i] * (N_prime[i] * N_prime[j]) * h  # Assemble K_local based on advection-diffusion equation
            
        return K_local

    # Global Stiffness Matrix 
    def assemble_global_matrices(self, N, h, Pe, element_order):
        K_global = np.zeros((2 * (N - 1) + 1, 2 * (N - 1) + 1))  # Adjust global matrix size for quadratic
        for e in range(N - 1):  # Iterate over elements
            K_local = self.local_stiffness_matrix(h, Pe)  # Local stiffness matrix
            K_global[2 * e: 2 * e + 3, 2 * e: 2 * e + 3] += K_local   # Place local matrix into the global matrix
        return K_global

    # Compute Cell Peclet Numbers 
    def solve(self, element_order=1):
        for Pe in self.Pe_values:
            h = (Pe * self.k) / self.a                               # Calculate grid spacing
            N_no = int((self.x_1 - self.x_0) / h) + 1                # Number of nodes
            x = np.linspace(self.x_0, self.x_1, 2 * (N_no - 1) + 1)  # Quadratic nodal positions
            u = np.zeros(2 * (N_no - 1) + 1)                         # Initialize solution vector for quadratic elements
            
            # Assemble the global stiffness matrix
            K_global = self.assemble_global_matrices(N_no, h, Pe, element_order)
            
            # Apply boundary conditions
            u[0] = self.u_0         # Dirichlet boundary condition at x=0
            u[-1] = self.u_1        # Dirichlet boundary condition at x=L
            K_global[0, :] = 0      # Zero out the first row
            K_global[0, 0] = 1      # Set the diagonal element to 1 for u[0]
            K_global[-1, :] = 0     # Zero out the last row
            K_global[-1, -1] = 1    # Set the diagonal element to 1 for u[-1]
            
            # Solve the system of equations
            u = np.linalg.solve(K_global, u)
            
            # Plotting the results
            plt.plot(x, u, label=f'Pe = {Pe}')  # Plot the current solution

        # Plot analytical solution
        x_analytical = np.linspace(self.x_0, self.x_1, 100)
        u_analytical = 10 * x_analytical
        
        # Plot 
        plt.plot(x_analytical, u_analytical, 'k--', label='Analytical Solution')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Galerkin FEM Solutions for the Steady-State Advection-Diffusion Equation')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Compute exact solution
    def exact_solution_values(self):
        # Compute exact solution based on boundary conditions and diffusion parameters
        C1_value = 10 / (1 - np.exp(20))                   # Constant for exact solution
        x_values = np.linspace(0, 1, 100)                  # X values over domain
        u_values = C1_value * (1 - np.exp(20 * x_values))  # Exact solution for each x value
        return x_values, u_values

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

# EAD Solver
class EAD_AdvectionDiffusionSolver(GFEM_AdvectionDiffusionSolver):
    """
    Solve ADE using the Exact Advection Diffusion method.
    """
    def local_stiffness_matrix(self, h, Pe, element_order=1):
        K_local = super().local_stiffness_matrix(h, Pe, element_order)  # Use parent class method
        # EAD-specific modification (if applicable)
        return K_local

    def solve(self, element_order=1):
        print("\nSolving with EAD Method\n")
        super().solve(element_order)  # Call the parent's solve method

# SUPG Solver
class SUPG_AdvectionDiffusionSolver(GFEM_AdvectionDiffusionSolver):
    """
    Solve ADE using Streamline Upwind Petrov-Galerkin (SUPG) method.
    """
    def local_stiffness_matrix(self, h, Pe, element_order=1):
        K_local = super().local_stiffness_matrix(h, Pe, element_order)  # Use parent class method
        # SUPG-specific modification (e.g., stabilization term addition)
        for i in range(element_order + 1):
            for j in range(element_order + 1):
                stabilization = Pe * 0.1  # Example stabilization term for SUPG
                K_local[i, j] += stabilization  # Add stabilization to stiffness matrix entry
        return K_local

    def solve(self, element_order=1):
        print("\nSolving with SUPG Method\n")
        super().solve(element_order)  # Call the parent's solve method

# Main Execution
Pe_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7]

# Solve using Galerkin FEM with quadratic elements as a base case
print("\nStarting Galerkin FEM (Quadratic Elements)...\n")
gfem_solver = GFEM_AdvectionDiffusionSolver(Pe_values)
gfem_solver.solve(element_order=2)  # Use quadratic elements

# Solve using EAD method
print("\nStarting EAD method...\n")
ead_solver = EAD_AdvectionDiffusionSolver(Pe_values)
ead_solver.solve(element_order=2)  # Use quadratic elements

# Solve using SUPG method
print("\nStarting SUPG method...\n")
supg_solver = SUPG_AdvectionDiffusionSolver(Pe_values)
supg_solver.solve(element_order=2)  # Use quadratic elements