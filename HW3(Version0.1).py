#!/usr/bin/env python3

import math 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, dsolve, exp, solve
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.polynomial import Polynomial

print("\nStarting Program...\n")

""" 
Nicolino Primavera 
FEM for Fluid Flow and FSI Interactions
Assignment 3 
11/27/24 

A damped harmonic oscillator is governed by (m * (d^2u / dt^2)) + c (du / dt) + ku = 0          # EQN 1
    - m is the object's mass
    - c is the damping coefficient 
    - k is the spring stiffness 
    - γ (gamma) = c / cr is the damping ratio 
    - cr = 2sqrt(mk) is the critical damping 
    - the system behaves as an undamped (γ = 0), underdamped (0 < γ < 1), overdamped (γ > 1), or critically damped (γ = 1) system 
    - ωn = sqrt(k / m) denotes the natural frequency of the system
    
Equation can be rewritten as (d^2u / dt^2) + 2 * γ * ωn * (du / dt) + (ωn^2) * u = 0            # EQN 2 
    - initial conditions u(0) = 1 , du/dt (0) = 1 , ωn = pi 

Solve EQN 2 for all the damping regimes (γ = 0, 0.5, 1, 2) using the generalized-α time integration method
    - For each γ, choose at least four values of the spectral radius parameters (0 ≤ ρ_∞ ≤ 1) and compare your numerical solution against the analytical solution
    - For each ρ_∞, refine the time step size, and comment on the error convergence and the numerical solution behavior 
"""

# Initialize initial conditions 
u_0 = 1                     # initial displacement 
dudt_0 = 1                  # initial velocity 
ωn = math.pi                # natural frequency 
m = 1 # example vlaue       # object mass 
k = ωn**2 * m               # spring stiffness
c_r = 2 * math.sqrt(m * k)  # critical damping 

# Define gammas - for all the damping regimes 
gamma = [0, 0.5, 1, 2]  # undamped, underdamped, critically damped, overdamped 

# Define spectral radius parameters 
rho_inf_values = [0, 0.25, 0.5, 1.0]    # spectral radius values

# Simulation parameters
time_steps = [0.1, 0.05, 0.01]  
t_end = 10.0 

# Analytical solution
def analytical_solution(gamma, t):
    """Analytical solution for the damped harmonic oscillator."""
    
    # Undamped case
    if gamma == 0:  
        return np.exp(-gamma * ωn * t) * (np.cos(ωn * t) + np.sin(ωn * t))
    
    # Underdamped case
    elif gamma < 1:  
        ωd = ωn * math.sqrt(1 - gamma**2)  # Damped natural frequency
        return np.exp(-gamma * ωn * t) * (np.cos(ωd * t) + np.sin(ωd * t))
    
    # Critically damped case
    elif gamma == 1:  
        return np.exp(-ωn * t) * (1 + ωn * t)
    
    # Overdamped case
    else:  
        λ1 = -ωn * (gamma - math.sqrt(gamma**2 - 1))
        λ2 = -ωn * (gamma + math.sqrt(gamma**2 - 1))
        return np.exp(λ1 * t) + np.exp(λ2 * t)

# Solve analytical solution  
t = np.linspace(0, 10, 100)  # Time from 0 to 10 seconds, 100 points

u_analytical_undamped = analytical_solution(gamma=0, t=t)       # undamped case 
u_analytical_underdamped = analytical_solution(gamma=0.5, t=t)  # underdamped case 
u_analytical_critical = analytical_solution(gamma=1, t=t)       # critically damped case
u_analytical_over = analytical_solution(gamma=2, t=t)           # over damped case 

#print(f"\nAnalytical solution for the undamped case: \n {u_analytical_undamped}\n")         # error handling 
#print(f"\nAnalytical solution for the underdamped case: \n {u_analytical_underdamped}\n")   # error handling 
#print(f"Analytical solution for the critically damped case:\n {u_analytical_critical}\n")   # error handling 
#print(f"Analytical solution for the overdamped case:\n {u_analytical_over}\n")              # error handling 

# Generalized-Alpha Time Integration Method 
def generalized_alpha_time_integraton_method():
    pass 

# Plot 
