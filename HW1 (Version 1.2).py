import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1          # advection speed
k = 0.05       # diffusion coefficient
Pe_list = [0.5, 1, 1.5, 2, 3, 5]  # Peclet numbers to explore

x_0, x_1 = 0, 1  # domain
u_0, u_1 = 0, 10  # boundary conditions

def gauss_quadrature_integration():
    # Use Gauss quadrature for numerical integration
    pass

def local_stiffness_matrix(Pe, h):
    # Compute local stiffness matrix using Gauss quadrature
    K_local = np.array([[1, -1], [-1, 1]]) * (1/h)  # Replace with actual calculation
    return K_local

for Pe in Pe_list:
    # Compute grid size based on Pe and define nodes
    h = Pe * k / a  # Grid size
    N_no = int((x_1 - x_0) / h) + 1
    x = np.linspace(x_0, x_1, N_no)

    # Initialize global matrices
    K_global = np.zeros((N_no, N_no))
    u = np.zeros(N_no)

    # Boundary conditions
    u[0], u[-1] = u_0, u_1

    # Assemble global stiffness matrix
    for e in range(N_no - 1):
        K_local = local_stiffness_matrix(Pe, h)
        K_global[e:e+2, e:e+2] += K_local

    # Apply boundary conditions to global matrix
    K_global[0, :] = 0
    K_global[0, 0] = 1
    K_global[-1, :] = 0
    K_global[-1, -1] = 1

    # Solve the system
    u = np.linalg.solve(K_global, u)

    # Plot the solution
    plt.plot(x, u, label=f'Pe = {Pe}')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()