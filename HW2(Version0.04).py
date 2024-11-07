#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, exp

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

# Cell Peclet Numbers
Pe_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7]   

# Solve for the exact solution 
def exact_solution_values():
    C1_value = 10 / (1 - np.exp(20))                    # C1 (constant) for exact solution
    x_values = np.linspace(0, 1, 100)                   # X values 
    u_values = C1_value * (1 - np.exp(20 * x_values))   # Exact solution in terms of x 
    return x_values, u_values

# Solve the analytical solution 
def analytical_solution(x_0, x_1):
    x_analytical = np.linspace(x_0, x_1, 100)
    u_analytical = 10 * x_analytical
    return x_analytical, u_analytical

# Galerkin FEM
class GFEM_AdvectionDiffusionSolver:

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
        print(f"\nSolving ADE using Galerkin FEM.\n")
        for Pe in self.Pe_values:
            h = (Pe * self.k) / self.a                               # Grid/Cell spacing - use uniform spacing
            N_no = int((self.x_1 - self.x_0) / h) + 1                # Computes number of nodes based on total domain length (x_1 - x_0)=1
            x = np.linspace(self.x_0, self.x_1, N_no)                # Computes nodal positions, N_no evenly spaced positions b/w the 1D domain boundaries
            K_global = self.assemble_global_matrices(N_no, h, Pe)    # Assembles the global stiffness matrix
            u = np.zeros(N_no)                                       # Initialize solution vector 'u' - stores the unknowns at each node
            
            # Apply boundary conditions
            u[0] = self.u_0         # Dirichlet BC at x=0
            u[-1] = self.u_1        # Dirichlet BC at x=L
            K_global[0, :] = 0      # Set the first row to 0
            K_global[0, 0] = 1      # Set the first diagonal element to 1
            K_global[-1, :] = 0     # Set the last row to 0
            K_global[-1, -1] = 1    # Set the last diagonal element to 1
            
            # Solve the linear system
            try:
                u = np.linalg.solve(K_global, u)                         # Solves the system of linear eqns K_global(u)=g
            except np.linalg.LinAlgError as e:
                print(f"Error solving linear system for Pe = {Pe}: {e}")
                continue
                
            # Plots the Pe numbers
            plt.plot(x, u, label=f'Pe = {Pe}')                       

        # Plot
        x_values, u_values = exact_solution_values()                           # Plot the exact solution
        plt.plot(x_values, u_values, label="Exact Solution", color="purple")   # Plot Peclet numbers 
        x_analytical, u_analytical = analytical_solution(self.x_0, self.x_1)   # Plot the analytical solution
        plt.plot(x_analytical, u_analytical, 'k--', label='Analytical Solution')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Galerkin FEM Solutions for the Steady-State Advection-Diffusion Equation')
        plt.legend()
        plt.grid(True)
        plt.show()

# Initialize and solve using Galerkin FEM
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

# Exact Advection-Diffusion (EAD) Method
class EAD_AdvectionDiffusionSolver(GFEM_AdvectionDiffusionSolver):

    # Initialize parameters (inheriting from the GFEM solver)
    def __init__(self, Pe_values, a=1.0, k=0.05, x_0=0, x_1=1, u_0=0, u_1=10):
        # Call the initializer of the GFEM solver class to set common parameters
        super().__init__(Pe_values, a, k, x_0, x_1, u_0, u_1)

    # Define the Local Stiffness Matrix for the EAD method (using linear elements) - same function as in GFEM
    def local_stiffness_matrix(self, h, Pe):
        K_local = np.zeros((2, 2))         # Initialize a 2x2 local stiffness matrix for linear elements
        xi, w = self.gauss_quadrature(2)   # Get Gauss quadrature points and weights for integration
        
        # Loop over the matrix rows (each row corresponds to a shape function)
        for i in range(2):
            # Loop over the matrix columns (each column corresponds to a shape function)
            for j in range(2):
                # Calculate the diffusion term, using 1 if i == j (diagonal) and -1 otherwise
                diffusion = (1 / h) * (1 if i == j else -1)
                
                # Calculate the advection term with weighting based on Gauss points
                # For node i, adjust the term based on Gauss quadrature location
                advection = (Pe * w[i] * (0.5 * (1 - xi[i]) if i == 0 else 0.5 * (1 + xi[i])))

                # Sum the diffusion and advection terms into the local stiffness matrix entry
                K_local[i, j] += diffusion + advection
        
        return K_local  # Return the computed 2x2 local stiffness matrix

    # Define the Local Stiffness Matrix for the EAD method using quadratic elements (higher-order)
    def local_stiffness_matrix_quadratic(self, h, Pe):
        K_local = np.zeros((3, 3))  # Initialize a 3x3 local stiffness matrix for quadratic elements
        xi, w = self.gauss_quadrature(3)  # Get Gauss quadrature points and weights for higher accuracy

        # Loop over rows for the quadratic local stiffness matrix
        for i in range(3):
            # Loop over columns for the quadratic local stiffness matrix
            for j in range(3):
                # Calculate the diffusion term, adjusting if i == j (diagonal) or not
                diffusion = (1 / h) * (1 if i == j else -1)

                # Compute the advection term for quadratic basis functions
                # The function `quadratic_basis_advection` provides the basis adjustment
                advection = (Pe * w[i] * self.quadratic_basis_advection(i, xi[i]))

                # Add both terms into the local matrix entry
                K_local[i, j] += diffusion + advection

        return K_local  # Return the computed 3x3 local stiffness matrix for quadratic elements

    # Helper method to define advection-related terms for quadratic basis functions 
    def quadratic_basis_advection(self, node_index, xi):
        # Define basis function contributions for each node index (0, 1, or 2)
        # These correspond to the standard quadratic shape functions
        if node_index == 0:
            return 0.5 * (xi**2 - xi)  # Shape function for first node (adjust as needed)
        elif node_index == 1:
            return 1 - xi**2           # Shape function for second (central) node
        elif node_index == 2:
            return 0.5 * (xi**2 + xi)  # Shape function for third node (adjust as needed)
    
    # Method to solve for each Peclet number using the EAD method for linear elements
    def solve_linear(self):
        print("\nSolving using EAD method with linear elements...\n")  
        results_linear = []
        for Pe in self.Pe_values:
            # Call the general solve method from the GFEM base class with linear elements
            u_values = super().solve()  # Replace this call with your FEM solver logic
            results_linear.append((Pe, u_values))  # Store result for this Pe value
        
        return results_linear

    # Method to solve for each Peclet number using the EAD method for quadratic elements
    def solve_quadratic(self):
        print("\nSolving using EAD method with quadratic elements...\n")  
        results_quadratic = []
        for Pe in self.Pe_values:
            # Adjust solver for quadratic elements, compute u_values (replace with FEM solution logic)
            u_values = super().solve()  # Replace this call with quadratic element solver logic
            results_quadratic.append((Pe, u_values))  # Store result for this Pe value
        
        return results_quadratic

    # Method to plot both linear and quadratic solutions against the exact solution
    def plot_solutions(self):
        x_values, u_exact_values = exact_solution_values()  # Get exact solution values

        # Plot exact solution for reference
        plt.plot(x_values, u_exact_values, label="Exact Solution", color="purple")

        # Solve and plot for each Pe using linear elements
        results_linear = self.solve_linear()
        for Pe, u_values in results_linear:
            plt.plot(x_values, u_values, label=f"Linear, Pe = {Pe}")

            # Finalize plot details
            plt.xlabel('x')     # Label the x-axis
            plt.ylabel('u(x)')  # Label the y-axis
            plt.title('Linear EAD Method Solutions vs. Exact Solution')  
            plt.legend()        # Add legend to the plot
            plt.grid(True)      # Enable grid for easier visualization
            plt.show()          # Display the plot

        # Solve and plot for each Pe using quadratic elements
        results_quadratic = self.solve_quadratic()
        for Pe, u_values in results_quadratic:
            plt.plot(x_values, u_values, '--', label=f"Quadratic, Pe = {Pe}")

            # Finalize plot details
            plt.xlabel('x')     # Label the x-axis
            plt.ylabel('u(x)')  # Label the y-axis
            plt.title('Quadratic EAD Method Solutions vs. Exact Solution')  
            plt.legend()        # Add legend to the plot
            plt.grid(True)      # Enable grid for easier visualization
            plt.show()          # Display the plot

# Solving using EAD method
solver_EAD = EAD_AdvectionDiffusionSolver(Pe_values)
solver_EAD.solve()

# Streamwise Upwind Petrov-Galerkin (SUPG) Method
class SUPG_AdvectionDiffusionSolver(GFEM_AdvectionDiffusionSolver):
    def __init__(self, Pe_values, a=1.0, k=0.05, x_0=0, x_1=1, u_0=0, u_1=10, tau=0.1):
        super().__init__(Pe_values, a, k, x_0, x_1, u_0, u_1)
        self.tau = tau  # Stabilization parameter for SUPG

    # Local Stiffness Matrix for SUPG method
    def local_stiffness_matrix(self, h, Pe):
        K_local = np.zeros((2, 2))
        xi, w = self.gauss_quadrature(2)
        advection_factor = Pe * self.a / h  # Scaling factor for advection

        for i in range(2):
            for j in range(2):
                diffusion = (1 / h) * (1 if i == j else -1)
                advection = advection_factor * w[i] * (0.5 * (1 - xi[i]) if i == 0 else 0.5 * (1 + xi[i]))

                # SUPG stabilization term
                residual = (1 if i == j else -1)
                stabilization = self.tau * residual * advection_factor
                
                K_local[i, j] += diffusion + advection + stabilization
        return K_local

    # Solving for each Pe with SUPG method
    def solve(self):
        print("\nSolving using SUPG method...\n")
        super().solve()  # Call the solve method from GFEM class

# Solving using SUPG method
solver_SUPG = SUPG_AdvectionDiffusionSolver(Pe_values)
#solver_SUPG.solve()            