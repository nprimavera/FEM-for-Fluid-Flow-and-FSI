from fenics import *
import numpy as np
import os
import pyvista 

print("\nWorking directory:", os.getcwd())
print("\nStarting program...\n")

# Create a unit square mesh and define temporal discretization
mesh = UnitSquareMesh(10, 10)
t = 0
T = 10
num_steps = 500
dt = T / num_steps

# Create two function spaces for velocity (quadratic) and pressure (linear)
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # Quadratic elements
Q = FunctionSpace(mesh, "Lagrange", 1)        # Linear elements
print("\nCreation of function spaces complete.\n")

# Create trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

## Dirichlet Boundary Conditions
# set u = 0 at the walls of the channel, that is at y = 0 and y = 1 
def walls(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[1], 1))
bc_noslip = DirichletBC(V, Constant((0, 0)), walls) # No-slip condition for velocity
# set p = 8 at the inflow (x = 0)
def inflow(x, on_boundary):
    return on_boundary and near(x[0], 0)
bc_inflow = DirichletBC(Q, Constant(8), inflow) # Pressure at inflow
# set p = 0 at the outflow (x = 1) - results in a pressure gradient that will accelerate the flow from the initial state with zero velocity 
def outflow(x, on_boundary):
    return on_boundary and near(x[0], 1)
bc_outflow = DirichletBC(Q, Constant(0), outflow)   # Pressure at outflow

# Collect BCs for the velocity and pressure in lists 
bcu = [bc_noslip]
bcp = [bc_inflow, bc_outflow]

print("\nBoundary Conditions complete.\n")

## Definition of the three variational forms - one for each step in the IPCS scheme
u_n = Function(V)
u_n.name = "u_n"
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)

# Constants
rho = Constant(1.0)
mu = Constant(1.0)
k = Constant(dt)
#f = Constant(mesh, Scalar(0, 0))

# Define Strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define Stress tensor 
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

# Define the variational formulation for the tentative velocity step
p_n = Function(Q, name="p_n")
p_n.assign(Constant(0.0))  # Initialize with zero

a1 = (rho * dot((u - u_n) / k, v) * dx
      + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
      + inner(sigma(U, p_n), epsilon(v)) * dx)
L1 = (- dot(Constant((0, 0)), v) * dx
      + dot(p_n * n, v) * ds
      - dot(mu * nabla_grad(U) * n, v) * ds)

# Assemble the bilinear form into the matrix A1
A1 = assemble(a1)
for bc in bcu:
    bc.apply(A1)

# Debug and print the assembled matrix
print("Matrix A1:\n", A1.array())

# Debugging terms in L1
try:
    term1 = assemble(-dot(Constant((0, 0)), v) * dx)
    term2 = assemble(dot(p_n * n, v) * ds)
    term3 = assemble(-dot(mu * nabla_grad(U) * n, v) * ds)
    print("L1 terms assembled successfully.")
except Exception as e:
    print("Error in L1 terms:", e)

# Assemble the full linear form L1
try:
    b1 = assemble(L1)
    print("Vector b1:\n", b1.get_local())
except Exception as e:
    print("\nError during assembly of L1:", e)



# Define the variational formulation for the tentative velocity step
p_n = Function(Q)
p_n.assign(Constant(0.0))  # Assign a zero field to start with
p_n.name = "p_n"

boundaries = MeshFunction("size_t", mesh, 1)  # Boundary markers
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)


a1 = (rho * dot((u - u_n) / k, v) * dx
      + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
      + inner(sigma(U, p_n), epsilon(v)) * dx)
L1 = (- dot(Constant((0, 0)), v) * dx
      + dot(p_n * n, v) * ds
      - dot(mu * nabla_grad(U) * n, v) * ds)

# Analytical solution
A1 = assemble(a1)  # Assemble the bilinear form into the matrix A1
for bc in bcu:     # Apply boundary conditions to the matrix
    bc.apply(A1)
print(A1)
try:
    b1 = assemble(L1)   # assemble L1 into a vector (linear form)
    print(b1.get_local())  # Print values of the vector
except Exception as e:
    print("\nError during assembly of L1:", e)

print("\nVariational problem for the tentative velocity step (Step 1) complete.")

# Variational formulation for the pressure correction step
u_ = Function(V)
a2 = (dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = (dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)

A2 = assemble(a2)
for bc in bcp:
    bc.apply(A2)
#try:
#    b2 = assemble(L2) 
#    print(b2)
#except Exception as e:
#    print("\nError during assembly of L2:", e)

print("\nVariational problem for the pressure correction step (Step 2) complete.")

# Variational formulation for the velocity correction step
p_ = Function(Q)
a3 = (rho * dot(u, v) * dx)
L3 = (rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)

A3 = assemble(a3)
#try:
#    b3 = assemble(L3)
#    print(b3)
#except Exception as e:
#    print("\nError during assembly of L3:", e)

print("\nVariational problem for the velocity correction step (Step 3) complete.\n")

# Prepare output files for velocity and pressure data
velocity_file = File("/Users/nicolinoprimavera/Desktop/Columbia University/Finite Element Method for Fluid Flow and Fluid-Structure Interactions/Final Project/velocity.pvd")
pressure_file = File("/Users/nicolinoprimavera/Desktop/Columbia University/Finite Element Method for Fluid Flow and Fluid-Structure Interactions/Final Project/pressure.pvd")

# Define the analytical solution as a UserExpression
class ExactSolution(UserExpression):
    def eval(self, values, x):              # This method defines the value of the exact solution at each point x
        values[0] = 4 * x[1] * (1.0 - x[1]) # Define the exact solution for the velocity component u (index 0)
        values[1] = 0.0                     # If the solution is 2D, the second component (index 1) is set to 0.0

    def value_shape(self):  # This method defines the shape of the values (2 components in the vector)
        return (2,)  # (2,) represents a 2-component vector

# Create the function that will store the interpolated exact solution
u_ex = Function(V)  # V is the vector function space for 2D velocity field - Interpolate the analytical solution into the function space V
u_exact = ExactSolution(degree=2)  # Create the exact solution object with polynomial degree 2
u_ex.interpolate(u_exact)  # Project the exact solution into the function space V

# Define the L² error variational formulation
L2_error = assemble(dot(u_ - u_ex, u_ - u_ex) * dx)  # squared difference between the numerical solution (u_) and the exact solution (u_ex)

print("\nAnalytical solution is interpolated into our function successfully and the L^2 error variational formulation is created.\n")

print("\nBeginning the time loop.\n")

# Create the loop over time
for i in range(num_steps):

    # Update current time step
    t += dt

    # Step 1: Tentative velocity step
    b1.vector()[:] = 0.0    # Set the initial values of b1 (right-hand side vector) to zero 
    assemble(L1, tensor=b1) # Assemble the vector b1 from the form L1
    b1.apply("insert")      # Apply the boundary conditions to the vector b1
    solve(A1, u_.vector(), b1)  # Solve the linear system for the tentative velocity u_
    u_.vector().apply("insert") # Update the velocity field

    # Step 2: Pressure correction step
    b2.vector()[:] = 0.0    # Set the initial values of b2 (right-hand side vector) to zero
    assemble(L2, tensor=b2) # Assemble the vector b2 from the form L2
    b2.apply("insert")      # Apply the boundary conditions to the vector b2
    solve(A2, p_.vector(), b2)  # Solve the linear system for the pressure field p_
    p_.vector().apply("insert") # Update the pressure field

    # Step 3: Velocity correction step
    b3.vector()[:] = 0.0    # Set the initial values of b3 (right-hand side vector) to zero
    assemble(L3, tensor=b3) # Assemble the vector b3 from the form L3
    b3.apply("insert")      # Apply the boundary conditions to the vector b3
    solve(A3, u_.vector(), b3)  # Solve the linear system for the corrected velocity u_
    u_.vector().apply("insert") # Update the velocity field with the correction
    
    # Update variable with solution from this time step
    u_n.vector()[:] = u_.vector()[:]
    p_n.vector()[:] = p_.vector()[:]

    # Write solutions to file (saving the velocity and pressure fields at each time step)
    velocity_file << u_
    pressure_file << p_

    # Compute error at the current time step
    error_L2 = np.sqrt(assemble(L2_error))
    error_max = np.max(np.abs(u_.vector().get_local() - u_ex.vector().get_local()))
    
    # Print error every 20th step and at the last step
    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")

print("\nTime loop complete.\n")

# Close file output for velocity and pressure
velocity_file.close()
pressure_file.close()

# Clean up: Destroy the right-hand side vectors and matrices to free memory
b1.clear()
b2.clear()
b3.clear()
A1.clear()
A2.clear()
A3.clear()

# Visualization (PyVista)
pyvista.start_xvfb()    # Start the X11 virtual framebuffer for rendering

# Create XDMF files for mesh and solution fields
mesh_file = File("results/mesh.xml")
velocity_file_vtk = File("results/velocity_solution.pvd")  # Changed file name to avoid overwriting
pressure_file_vtk = File("results/pressure_solution.pvd")  # Changed file name to avoid overwriting

# Write the mesh to a VTK-compatible file (in XML format)
mesh_file << mesh

# Write the solution (velocity and pressure) to corresponding VTK files
velocity_file_vtk << u_
pressure_file_vtk << p_

# Load the mesh and solutions into PyVista for visualization
mesh_vtk = pyvista.read("results/mesh.xml")
velocity_vtk = pyvista.read("results/velocity_solution.pvd")
pressure_vtk = pyvista.read("results/pressure_solution.pvd")

# Create glyphs for visualizing the velocity field using arrows
glyphs = velocity_vtk.glyph(orient="vectors", scale="magnitude", factor=0.1)

# Set up the plotter
plotter = pyvista.Plotter()
plotter.add_mesh(mesh_vtk, style="wireframe", color="k")  # Show mesh in wireframe
plotter.add_mesh(glyphs)  # Add velocity glyphs to mesh
plotter.view_xy()

# Display the plot or save as an image
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")