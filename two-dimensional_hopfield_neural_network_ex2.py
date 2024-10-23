import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Activation function (tanh)
def sigma(x):
    return np.tanh(x)


# The system of differential equations for the Hopfield network
def hopfield_system(t, x, w):
    W = np.array([[0, w], [w, 0]])  # Weight matrix
    I = np.array([0, 0])  # External input vector
    return -x + W @ sigma(x) + I


# Task 1: Analyze stability at the equilibrium point (0, 0)
def analyze_stability(w):
    print(f"Analyzing stability for w = {w}")
    J = np.array([[-1, w], [w, -1]])  # Jacobian at (0, 0)
    eigenvalues = np.linalg.eigvals(J)
    print(f"Eigenvalues for w = {w}: {eigenvalues}")
    if np.all(np.real(eigenvalues) < 0):
        print("The system is stable at (0, 0)")
    else:
        print("The system is unstable at (0, 0)")


# Task 2: Find equilibrium points for w = 0.5, 1, 2
def find_equilibrium_points(w):
    print(f"\nFinding equilibrium points for w = {w}")
    equilibrium_points = np.array([[0, 0]])  # (0, 0) is the trivial equilibrium point
    print(f"Equilibrium point: (0, 0)")

    # Analyzing stability at (0, 0)
    analyze_stability(w)


# Task 3: Simulate the system and plot direction fields and trajectories
def simulate_system(w, initial_conditions):
    print(f"\nSimulating the system for w = {w}")

    # Solve the system using solve_ivp
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 500)

    fig, ax = plt.subplots()
    for x0 in initial_conditions:
        sol = solve_ivp(hopfield_system, t_span, x0, args=(w,), t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1], label=f"x0 = {x0}")

    # Plotting direction field
    x1_vals = np.linspace(-2, 2, 20)
    x2_vals = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    U = -X1 + w * np.tanh(X2)
    V = -X2 + w * np.tanh(X1)
    ax.quiver(X1, X2, U, V, color='gray', alpha=0.6)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Direction Field and Trajectories for w = {w}')
    ax.legend()
    plt.show()


# Main function with a task selection menu
def main():
    initial_conditions = [[0.5, 0], [1, 1], [-1, -1]]  # Different initial conditions for simulation

    while True:
        print("\nChoose a task to execute:")
        print("1: Analyze stability of the equilibrium point (0, 0)")
        print("2: Find equilibrium points and analyze stability")
        print("3: Simulate system and plot direction fields and trajectories")
        print("0: Exit")
        choice = input("Enter your choice (0-3): ")

        if choice == '1':
            w = float(input("Enter the value of w: "))
            analyze_stability(w)

        elif choice == '2':
            for w in [0.5, 1, 2]:
                find_equilibrium_points(w)

        elif choice == '3':
            w = float(input("Enter the value of w: "))
            simulate_system(w, initial_conditions)

        elif choice == '0':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please enter a number between 0 and 3.")


# Run the program
if __name__ == "__main__":
    main()
