# Initialize channel parameters
H = 1           # height of the channel (cm)
L = 8           # length of the channel (cm)
print(f"Channel dimensions: {L} cm x {H} cm")

# Fluid parameters
ρ = 1           # fluid density (g/cm^3)
μ = 1           # fluid viscosity (g/cm/s)
print(f"Fluid parameters: ρ = {ρ} g/cm^3, μ = {μ} g/cm/s\n")

# Create a rectangular mesh with a specified number of elements
nx, ny = 80, 10  # Elements along x and y directions
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

# Step 1: Define function spaces for P1P1 elements
V_P1 = VectorFunctionSpace(mesh, "P", 1)  # Velocity space (linear elements)
Q_P1 = FunctionSpace(mesh, "P", 1)       # Pressure space (linear elements)
W_P1 = FunctionSpace(mesh, MixedElement([V_P1.ufl_element(), Q_P1.ufl_element()]))

# Step 2: Trial and test functions
(u, p) = TrialFunctions(W_P1)
(v, q) = TestFunctions(W_P1)

# Step 3: Stabilization parameter
h = mesh.hmin()  # Minimum mesh size
tau = Constant(1 * h / μ)  # Stabilization parameter

# Step 4: Define bilinear and linear forms with stabilization
a_P1 = (
    μ * inner(grad(u), grad(v)) * dx
    - div(v) * p * dx
    - div(u) * q * dx
    + tau * div(u) * div(v) * dx  # Stabilization for continuity
    + tau * p * q * dx            # Pressure stabilization
)

L_P1 = dot(Constant((0.0, 0.0)), v) * dx

# Refine the mesh
nx, ny = 320, 80  # Further refined mesh
mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

# Step 5: Apply boundary conditions
bcs_P1 = [
    DirichletBC(W_P1.sub(0), U_in, "near(x[0], 0)"),  # Inflow
    DirichletBC(W_P1.sub(0), Constant((0.0, 0.0)),   # No-slip walls
                "on_boundary && (near(x[1], 0) || near(x[1], {H}))".format(H=H)),
    DirichletBC(W_P1.sub(1), Constant(0.0), "near(x[0], 0)")  # Pressure reference
]

# Step 6: Solve the system
w_P1 = Function(W_P1)

# Enable verbose solver output for debugging
PETScOptions.set("ksp_type", "cg")  # Conjugate gradient method
PETScOptions.set("pc_type", "ilu")  # Incomplete LU preconditioning
PETScOptions.set("ksp_monitor")  # Enable solver monitoring

# Assemble and solve
solve(a_P1 == L_P1, w_P1, bcs_P1)

# Step 7: Extract velocity and pressure solutions
(u_P1, p_P1) = w_P1.split()

# Compute analytical pressure gradient
U_max = 1.0  # Maximum velocity at centerline (cm/s)
analytical_pressure_gradient = -2 * μ * U_max / (H**2)
print(f"Analytical Pressure Gradient: {analytical_pressure_gradient}")

# Compute the numerical pressure gradient
pressure_at_start = p_P1(Point(float(0), float(H / 2)))  # Start of the centerline
pressure_at_end = p_P1(Point(float(L), float(H / 2)))   # End of the centerline
print(f"\nPressure at start (x=0): {pressure_at_start}")
print(f"\nPressure at end (x=L): {pressure_at_end}")
print(f"\nChannel Length (L): {L}")

# Compute pressure gradient
pressure_gradient_P1 = (pressure_at_end - pressure_at_start) / L
print(f"\nNumerical Pressure Gradient (P1P1): {pressure_gradient_P1}")

# Compute relative error compared to the analytical gradient
relative_error_P1 = abs(
    (pressure_gradient_P1 - analytical_pressure_gradient) / analytical_pressure_gradient
) * 100
print(f"\nRelative Error (P1P1): {relative_error_P1:.2f}%")

# Step 8: Check pressure values
p_values = p_P1.vector().get_local()
print("\nMin pressure:", np.min(p_values))
print("Max pressure:", np.max(p_values))
print("Any NaNs:", np.any(np.isnan(p_values)))
print("Any Infs:", np.any(np.isinf(p_values)))

# Step 9: Plot results if values are valid
if not np.any(np.isnan(p_values)) and not np.any(np.isinf(p_values)):
    plot(u_P1, title="Velocity Field (P1P1)")
    plot(p_P1, title="Pressure Field (P1P1)")

    # Compute and compare pressure gradient
    pressure_gradient_P1 = (
        p_P1(Point(x_coords[-1], H/2)) - p_P1(Point(x_coords[0], H/2))
    ) / L
    print(f"\nNumerical Pressure Gradient (P1P1): {pressure_gradient_P1}")
    relative_error_P1 = abs(
        (pressure_gradient_P1 - analytical_pressure_gradient)
        / analytical_pressure_gradient
    ) * 100
    #print(f"\nRelative Error (P1P1): {relative_error_P1:.2f}%\n")
else:
    print("\nInvalid pressure values; check the solver or stabilization setup.")

