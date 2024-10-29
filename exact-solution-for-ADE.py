import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, exp, solve

"""
Nicolino Primavera
FEM for Fluid Flow and FSI
10/29/24

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

# Plot the exact solution
plt.figure(figsize=(8, 6))
plt.plot(x_values, u_values, label=r"$u(x) = 10 / (1 - np.exp(20)) * (1 - e^{20x})$" + f" ", color="purple")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Exact Solution of the Steady-State Advection-Diffusion Equation with Constants")
plt.legend()
plt.grid(True)
plt.show()