#!/usr/bin/env python3

# FEM for Fluid Flow and FSI Interractions
# Assignment 1 
# Using the Finite Element Method (FEM) to analyze the Steady-state Advection-Diffusion Equation
# Steady-state Advection-Diffusion Equation: (du/dx) = (0.05)*(d^2u/dx^2) - strong form, Domain: 0 <= x <= 1 , u(0) = 0 & u(1) = 10

import math 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Want to explore the effect of cell Peclet number on the computed solution when Galerkin FEM is applied to this problem
# Use numerical inegration or Gauss quadrature-based integration (Gauss integration)

# Notes 
    # Convection & diffusion are not properly balanced in the numerical scheme (Galerkin method) --> numerical stability issues
    # Results in oscillations in u^h
    # Galerkin form: a(w^h , u^h) = (w^h , f)
    # (u - u^h) is the error -- error is orthogonal to the test function space , space: Galerkin orthogonality 
    # exact solution (u) also satisfies discrete approximation (u^h) --> "consistent"
    # consistent/Galerkin orthogonality + stability --> guarantees optimal convergence 
    # convection dominated flows - Galerkin method is unstable, but it is consistent and satisfies Galerkin orthogonality 
    # Pe number affects the accuracy of the Galerkin method --> large numbers are unstable (numerical oscillations)

# a. Choose at least four different grids for a range of cell Peclet numbers (Pe) and demonstrate that oscillatory solutions are obtained when Pe > 2

#Pe = (a*h)/k = 20 * h            # cell peclet number , Pe ~ convection/diffusion
a = 1                             # advection/convection speed
k = 0.05                          # diffusion coefficient 

Pe = [0.5, 1, 1.5, 2, 4, 5]    # choose at least four different grids for a range of Pe numbers, choose some that are about 2, above 2 and below 2
#h = ((Pe*k)/a)                    # grid/cell spacing - use uniform spacing 

# Define domain 
x_0 = 0
x_1 = 1

# Handling the Peclet Number 
for pe in Pe:
    # Update the grid size and stiffness matrix for each Pe & corresponding new h
    h = (pe * k) / a  

    # Define number of nodes & elements 
    N_no = int(((x_1 - x_0)/h) + 1)   # number of nodes (linear) - for quadratic mesh N_no_q = N_no_l + N_el - every element will get a midpoint 
    N_el = N_no - 1                   # number of elements 

    # Initialize arrays 
    x = np.linspace(x_0, x_1, N_no)   # position coordinates
    K = np.zeros((N_no, N_no))        # global stiffness matrix
    u = np.zeros(N_no)                # unknowns vector

    # Define a vector of unknowns that we want to solve and a global K matrix [N_no x N_no] - size
    #x[N_no]                           # x vector - "global x" or "position coordinates"
    #K[N_no , N_no]                    # "global k matrix" with sizing [N_no x N_no]
    #u[N_no]                           # u vector - "unknowns"

    # Applying boundary conditions: u(0) = 0 and u(1) = 10
    u[0] = 0                          # u at x = 0
    u[-1] = 10                        # u at x = 1

    # Assemble local stiffness matrix  

    # Loop over each element from e = 1, ... , N_el
        # Define your local coordinates 
        # Assume a linear basis function - define what your shape functions are locally
        # quadratic has three shape functions per element  
    for e in range(N_el):
        # Local stiffness matrix for element e
        K_local = np.array([[1, -1], 
                            [-1, 1]]) * (1/h)  # linear elements

        # Global assembly (place K_local into global K)
        K[e:e+2, e:e+2] += K_local

    # Apply BCs to K
    K[0, :] = 0
    K[0, 0] = 1
    K[-1, :] = 0
    K[-1, -1] = 1

    # Solve the system once matrix assembly is done and BCs are applied 
    u = np.linalg.solve(K, u)  # solve the system Ku = F

    # Plot 
    plt.plot(x, u, label=f'Pe = {pe}')

# Visualization and Comparison
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()



#nodes = [1, 2, 3, 4, ... , N_no]  # A's nodes / node number - conditional boundry values at 1 & N_no 
#elements = [x_A , x_A+1]          # elements - conveniently b/w nodes 


#eNoN = 2                          # NoN = number of nodes , setting it to 2, but it varies depending on the level of accuracy that you want 
#N_a[eNoN]                         # shape function - vector depending on eNoN

# if you use integration points / Gauss Quadrature Rule, you need to define the shape function at each quadrature
def gauss_quadrature_points(n):
    """ Return quadrature points and weights for n-point Gauss quadrature. """
    if n == 2:
        # Two-point Gauss quadrature
        return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1, 1])
    # Add more points for higher order integration

# Compute your Jacobian 
#X,e                               # Jacobian 
#N_a,e[eNoN, 1]                    # Jacobian shape function - vector

# Constructing your stiffness matrix - loop through each element (e = 1, N_el) and construct local stiffness matrix (k^e)_ab
#K_local = [K_11, K_12,  
#           K_21, K_22]            # Construction step - construct each entry of the local stiffness matrix - [eNoN x eNoN] --> [2 x 2] or [3 x 3]

# Assembly function - create a mapping for every node number (they interact w/ their neighbors)
    # For each entry of K_ab we have to map its corresponding global node number 
    # K_ab - a would be mapped to 'A' and b to 'A+1'
    # Add - K_AB (global matrix) should be replaced by K_AB (previous value) + K_ab (new value)
    # key is to find the right mapping to assemble the whole matrix 

# Whole Matrix - (Symmetric Matrix [N_no x N_el] * u [N_no x 1] = g)
    # Apply BCs to the symmetric matrix - replace the first and last value with 1, and make the rest of the first and last row 0
    # Set the RHS to the boundary condition (ex. u_1 = g_1 or u_N_no = g_N) - allows us to set the condition we want equal to g_1
    # BCs force the problem 
    # will get a trivial solution (ex. 0) or a random solution if we don't do this 
    # Normally the RHS is 0 of this problem because there is no forcing 
    # Can divide the diagonal of the matrix by a K_max to try and converge faster 

# Another way to solve is using the weak form a(w^h , v^h) = (w^h , g^h) --> will need to integrate again S Na , g^h

# b. Compare the solutions for Pe < 2 against the analytical solution 

# Finite differences --> Central Differencing (CD) --> leads to stability restrictions 
    # Pe < 2 for stability 