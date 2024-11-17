#!/usr/bin/env python3

import math 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, exp, solve
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.polynomial import Polynomial

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

# Parameters
a = 1.0                                                # Advection speed
k = 0.05                                               # Diffusion coefficient
#Pe = (a*h)/k = 20*h                                   # Cell peclet number (formula)
Pe_values = [0.1, 1, 2, 3, 7]                          # Peclet numbers 
#h = (Pe*k)/a                                          # Grid/cell spacing - use uniform spacing 

# Domain
x_0 = 0 
x_1 = 1

# Boundary Conditions
u_0 = 0
u_1 = 10  

# Solve for the exact solution 
def exact_solution():
    """
    This code solves the Advection-Diffusion Equation: du/dx = 0.05 * d^2u/dx^2
    Solves the ordinary differential equation, du/dx = 0.05 * d^2u/dx^2, and plots the exact solution
    """

    # Define symbolic variables and function for solving the ODE
    x_sym = symbols('x')
    u_sym = Function('u')(x_sym)

    # Define the differential equation: du/dx = 0.05 * d^2u/dx^2, rewritten as 0.05 * d^2u/dx^2 - du/dx = 0
    ODE = Eq(0.05 * u_sym.diff(x_sym, x_sym) - u_sym.diff(x_sym), 0)

    # Solve the differential equation - by hand you get 0 and 20 with u(x)=C1+C2e^20x as the general solution
    general_solution = dsolve(ODE, u_sym)   # dsolve is an ODE solver 

    # Applying boundary conditions u(0) = 0 and u(1) = 10 to solve for constants C1 and C2
    C1, C2 = symbols('C1 C2')
    boundary_conditions = {u_sym.subs(x_sym, 0): 0, u_sym.subs(x_sym, 1): 10}

    # Substitute boundary conditions into general solution to find C1 and C2
    u_general = general_solution.rhs
    C1_value = 10 / (1 - np.exp(20))  # Derived value for C1 for x=1
    C2_value = -C1_value              # Since C2 = -C1 for x=0

    # Define the exact solution function using C1 and C2
    def exact_solution(x):
        return C1_value * (1 - np.exp(20 * x))  # Exact Solution in terms of x

    # Generate x and u(x) values for plotting
    x_values = np.linspace(0, 1, 100)
    u_values = exact_solution(x_values)
    return x_values, u_values

# Solve the analytical solution 
def analytical_solution(x_0, x_1):
    x_analytical = np.linspace(x_0, x_1, 100)
    u_analytical = 10 * x_analytical
    return x_analytical, u_analytical

"""Solving using Galerkin FEM"""

# Gauss Quadrature / Numerical Integration 
def gauss_quadrature(n_points):
    """
    Returns Gauss quadrature points and weights for numerical integration
    Gauss Quadrature Formula: Integral from -1 to +1 of (f(ξ)*dξ) = Summation from g=1 to # of int points (n) of f(ξ_g)*w_g
        f(ξ) - function being integrated 
        ξ_g - quadrature points                               
        w_g - weights associated with each quadrature point , determine how much each function evaluation contributes to total integral
        n - number of quadrature points 

    2 point Gauss Quadrature:
        Quadrature points: ξ_1 = -1/sqrt(3) , ξ_2 = 1/sqrt(3) - these points are extremely accurate for linear and quadratic basis functions
        Weights: w_1 = w_2 = 1 
        Use to integrate polynomials up to 3rd degree 

    Use quadrature points to integrate local stiffness matrix contributions within each element by evaluating the shape functions & their derivatives at the quadrature points
    Increase the number of quadrature points to improve accuracy of if integrating a more complex function
    """
    # Gauss Quadrature Rule for Numerical Integration 
    if n_points == 2:                                                       # 2-point quadrature 
        return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1, 1])    # Quadrature points and weights 
    else:
        raise ValueError("Only 2-point quadrature implemented.")

# Stiffness Matrix 
def local_stiffness_matrix(h, Pe):
    """
    Compute the local stiffness matrix for a finite element of size h and Peclet number Pe in a 1D advection-diffusion problem
    
    Compute the Diffusion Term for each matrix entry:
        1/h - derivatives of the shape functions wrt physical coordinates 
        diffusion term is proportional to the inverse of the element size 
        when i == j the value is 1 (diagonal)
        when 1 /= j the value is -1 (off-diagonal)

    Compute the Advection Term for each matrix entry:
        Pe - peclet number scales the contribution of advection relative to diffusion 
        w[i] - weight associated w/ Gauss quadrature point i 
        (0.5 * (1 - xi[i]) - shape function N_1(ξ) evaluated at quadrature point i
        (0.5 * (1 + xi[i]) - shape function N_2(ξ) evaluated at quadrature point i
    """
    K_local = np.zeros((2, 2))                                               # Initialize Local Stiffness Matrix [2 x 2]
    #print(f"\nInitial Local Stiffness Matrix (should be 0): \n{K_local}\n") # Error handling 
    
    xi, w = gauss_quadrature(2)                                              # 2-point Gauss quadrature points (xi) and weights (w) 
    #print(f"Gauss Quadrature: {xi, w}")                                     # Error handling 
    #print(f"Quadrature points: {xi}")                                       # Error handling 
    #print(f"Weights: {w}\n")                                                # Error handling 

    # Nested loop to fill the Local Stiffness Matrix [2 x 2]
    #print("\nCalculating Local Stiffness Matrices: \n")
    iterations = 0      # initialize counter
    for i in range(2):
        for j in range(2):
            # Diffusion term (du/dx) (constant over the element for linear basis functions)
            diffusion = (1/h) * (1 if i == j else -1)                        # Computes the Diffusion Term
            #print(f"Diffusion Term: {diffusion}")                           # Error handling

            # Advection term (Pe * u * v)
            advection = Pe * w[i] * (0.5 * (1 - xi[i]) if i == 0 else 0.5 * (1 + xi[i]))    # Computes the Advection Term 
            #print(f"Advection Term: {advection}")                                          # Error handling 

            # Increment the counter
            iterations +=1 

            # Local Stiffness Matrix
            K_local[i, j] += diffusion + advection                                                         # Computes the Local Stiffness Matrix
            #print (f"Local Stiffness Matrix after {iterations} iterations: \n{K_local}\n")                # Error handling
    
    return K_local 

# Global Stiffness Matrix 
def assemble_global_matrices(N, h, Pe):
    """
    The Global Stiffness Matrix represents the entire system of equations for all nodes in the domain [0,1[ - comprehensive assembly of contributions from each element's local stiffness matrix 
    Assembly: The local stiffness matrix for each element is computed and assembled into the global stiffness matrix by adding values to the corresponding rows and columns (nodes) of the global matrix
    The Global Stiffness Matrix combines all the local element stiffness matrices, which creates a global system of equations that can be solved to approx the soln of the advection-diffusion equation over the entire domain
    Compute the Global Stiffness Matrix for the domain [0,1] with N nodes

    N - number of nodes in the finite element mesh
    h - size of each element (distance b/w adjacent nodes), uniform for a structured grid 
    Pe - peclet number 
    """
    K_global = np.zeros((N, N))                                             # Initialize an empty array of size [N x N] (nodes)
    #print(f"\nGlobal stiffness matrix (empty): \n{K_global}\n")            # Error handling - empty array of size [N x N] - will be one for each Pe number (11)
    
    # Loops over elements in the domain 
    for e in range(N-1):                                                    # Elements are iterated N-1 b/c there is one element b/w every two nodes, element number (e) goes from 0 to N-2
        # Local stiffness matrix
        K_local = local_stiffness_matrix(h, Pe)                             # Assembling the local stiffness matrix [2 x 2] for each element - represents the interactions b/w the two nodes that define the element
        #print(f"Local Stiffness Matrix for element {e}: \n{K_local}\n")    # Error handling 
        
        # Global stiffness matrix - assembling the local stiffness matrices into the corresponding part of the global stiffness matrix
        K_global[e:e+2, e:e+2] += K_local                                   # Indexing: for element e, the nodes involved are node e and node e+1 --> K_global[e:e+2, e:e+2] refers to the 2 x 2 submatrix of K_global that corresponds to these two nodes
        #print(f"Global Stiffness Matrix for element {e}: \n{K_global}\n")  # Error handling 

    # Print total number of elements used for each global stiffness matrix
    #print(f"Total number of elements used for Peclet number {Pe}: {e}")
    
    return K_global                                                         # Return the Global stiffness matrix 

# Solving each Cell Peclet Number 
print(f"\nSolving the ADE using Galerkin FEM.\n")
for Pe in Pe_values:                                    # Iterate over selected Peclet numbers from above array 
    # Compute grid size based on Pe and define nodes
    h = (Pe * k) / a                                    # Calculates grid size based on Pe
    N_no = int((x_1 - x_0) / h) + 1                     # Computes number of nodes based on total domain length (x_1 - x_0)=1
    x = np.linspace(x_0, x_1, N_no)                     # Generates nodes positions , N_no evenly spaced points b/w the domain boundaries --> positions of the nodes in the 1D domain
    #print(f"Grid size for Peclet number {Pe}: {h}\n")                          # Error handling 
    #print(f"Number of nodes for Peclet number {Pe}: {N_no}\n")                 # Error handling 
    #print(f"Nodal positions for Peclet number {Pe}: \n {x}\n")                 # Error handling 

    # Initialize global matrices
    K_global = assemble_global_matrices(N_no, h, Pe)    # Assembles the Global Stiffness Matrix from the local element matrices
    u = np.zeros(N_no)                                  # Initialize solution vector 'u' - stores the unknowns at each node
    #print(f"Global Stiffness Matrix for Peclet number {Pe}: \n{K_global}\n")   # Error handling 
    #print(f"Solution vector 'u' for Peclet number {Pe}: \n{u}\n")              # Error handling - should be all zeros

    # Apply boundary conditions
    u[0] = u_0                                          # Sets first value of u at x=0 to the BC 0
    u[-1] = u_1                                         # Sets final value of u at x=1 to th BC 10
    
    # Modify Global Stiffness Matrix to enforce BCs
    K_global[0, :] = 0                                  # Clears the first row            
    K_global[0, 0] = 1                                  # Sets the first diagonal element to 1
    K_global[-1, :] = 0                                 # Clears the last row 
    K_global[-1, -1] = 1                                # Sets the last diagonal element to 1 

    # Solve the linear system
    try:
        u = np.linalg.solve(K_global, u)                      # Solves the system of linear eqns K_global(u)=g
        #print(f"Solution vector 'u' for Peclet number {Pe}: \n{u}\n")           # Error handling 
    except np.linalg.LinAlgError as e:
        print(f"Error solving linear system for Pe = {Pe}: {e}")
        continue

    # Plot
    plt.plot(x, u, label=f'Pe = {Pe}')

# Plot the Pe numbers against the analytical solution 
x_analytical, u_analytical = analytical_solution(x_0, x_1)   # Plot the analytical solution
plt.plot(x_analytical, u_analytical, 'k--', label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Galerkin FEM Solutions for the Steady-State Advection-Diffusion Equation')
plt.legend()
plt.grid(True)
plt.show()

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

# Gauss Quadrature
def gauss_quadrature(n_points):
    if n_points == 3:
        xi = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
        return xi, w
    else:
        raise ValueError("Only 3-point quadrature implemented.")

# Define higher-order (quadratic) basis functions for EAD and SUPG methods
def quadratic_shape_functions(xi): 
    N = np.array([0.5 * xi * (xi - 1), (1 - xi**2), 0.5 * xi * (xi + 1)])       # Shape functions
    #print(f"\nQuadratic shape functions:\n {N}\n")
    dN_dxi = np.array([xi - 0.5, -2 * xi, xi + 0.5])                            # Derivatives with respect to xi
    #print(f"\nDerivatives of shape functions:\n {dN_dxi}\n")
    return N, dN_dxi    # Returns the shape functions and their derivatives 

# Compute the local stiffness matrix for quadratic elements using EAD or SUPG methods
def local_stiffness_matrix_quadratic(h, Pe, method="Galerkin"):
    """
    `method` argument specifies whether to use "Galerkin", "EAD", or "SUPG" stabilization - computed below
    """
    K_local = np.zeros((3, 3))      # Initializing local stiffness matrix
    xi, w = gauss_quadrature(3)     # Quadrature points and weights 
    
    for i in range(3):                  # Rows 
        for j in range(3):              # Columns
            for p in range(len(xi)):    # Quadratic shape functions 
                 
                N, dN_dxi = quadratic_shape_functions(xi[p])        # Quadratic shape functions and derivatives
                dN_dx = dN_dxi / (0.5 * h)                          # Derivatives wrt physical coordinates               
                diffusion = (k / h) * dN_dx[i] * dN_dx[j] * w[p]    # Diffusion term
                
                advection = a * N[i] * dN_dx[j] * w[p]              # Advection term with SUPG stabilization

                # Add stabilization term if SUPG is selected
                if method == "SUPG" and Pe > 2:
                    tau_SUPG = 0.5 * h / a                          # SUPG stabilization parameter
                    stabilization = tau_SUPG * dN_dx[i] * dN_dx[j]  # SUPG stabilization term
                else:
                    stabilization = 0
                
                K_local[i, j] += diffusion + advection + stabilization  # Compute local stiffness matrix 
    #print(f"\nLocal stiffness matrix:\n {K_local}\n")
    return K_local

# Define load vector for quadratic elements
def load_vector_quadratic(h):
    xi, w = gauss_quadrature(3)     # Gauss quadrature points and weights 
    f_local = np.zeros(3)           # Initialize load vector 
    for i in range(3):              # Rows 
        for p in range(len(xi)):    # Shape functions 

            N, _ = quadratic_shape_functions(xi[p])     # Shape functions and derivatives 
            f_local[i] += N[i] * w[p] * h               # Load term
    #print(f"\nLocal load vector:\n {f_local}\n")
    return f_local

# Assemble global matrices for quadratic elements
def assemble_global_matrices_quadratic(N, h, Pe, method="Galerkin"):
    
    K_global = np.zeros((2 * N - 1, 2 * N - 1))     # Initialize global stiffness matrix
    #print(f"\nGlobal stiffness matrix:\n {K_global}\n")
    F_global = np.zeros(2 * N - 1)                  # Initialize global load vector
    #print(f"\nGlobal load vector:\n {F_global}\n")
    
    for e in range(N-1):    
        K_local = local_stiffness_matrix_quadratic(h, Pe, method)   # Compute local stiffness matrix for current element  
        f_local = load_vector_quadratic(h)                          # Compute local load vector for current element 
        
        # Map local to global nodes for quadratic elements
        global_nodes = [2*e, 2*e + 1, 2*e + 2]  # Each quadratic element has 3 associated nodes in the global system
        
        for i in range(3):          # Row
            for j in range(3):      # Column 
                
                K_global[global_nodes[i], global_nodes[j]] += K_local[i, j]     # Assembles the global stiffness matrix 
            
            F_global[global_nodes[i]] += f_local[i]     # Assembles the global load vector 
    
    #print(f"\nGlobal stiffness matrix:\n {K_global}\n , \nGlobal load vector:\n {F_global}\n")
    return K_global, F_global

# Run analysis for each method
methods = ["EAD", "SUPG", "Galerkin"]
for method in methods:
    print(f"\nSolving the ADE using {method}.\n")
    for Pe in Pe_values:

        h = (Pe * k) / a                            # Grid size 
        N_no = int((x_1 - x_0) / h) + 1             # Number of nodes
        x = np.linspace(x_0, x_1, 2 * N_no - 1)     # Nodal positions 
        
        # Assemble global matrices and apply boundary conditions
        K_global, F_global = assemble_global_matrices_quadratic(N_no, h, Pe, method)
        
        # Apply boundary conditions
        F_global[0] = u_0
        F_global[-1] = u_1
        K_global[0, :] = K_global[-1, :] = 0
        K_global[0, 0] = K_global[-1, -1] = 1
        
        # Solve for u
        try:
            u = np.linalg.solve(K_global, F_global)
            #u = np.linalg.solve(K_global, u)
            #print(f"Solution vector 'u' for Peclet number {Pe}: \n{u}\n")           # Error handling 
        # Solve the linear system
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system for Pe = {Pe}: {e}")
            continue
        
        # Plot solution for each method
        plt.plot(x, u, label=f'{method}, Pe = {Pe}')

        
# Plot EAD and SUPG against exact solution
x_values, u_values = exact_solution()    # Exact solution for comparison
plt.plot(x_values, u_values, label='Exact Solution', color='purple')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('EAD and SUPG Solutions for the Steady-State Advection-Diffusion Equation')
plt.legend()
plt.grid(True)
plt.show()