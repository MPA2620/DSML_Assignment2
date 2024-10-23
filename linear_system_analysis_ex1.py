import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_eigenvalues_eigenvectors():
    A = np.array([[-1, -2, 1],
                  [2, -1, 1],
                  [0, 0, -3]])

    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)


def determine_stability():
    A = np.array([[-1, -2, 1],
                  [2, -1, 1],
                  [0, 0, -3]])
    eigenvalues, _ = np.linalg.eig(A)

    stable = all(np.real(eigenvalue) < 0 for eigenvalue in eigenvalues)
    if stable:
        print("The system is stable at x = 0.")
    else:
        print("The system is not stable at x = 0.")


def solve_linear_system():
    def system(t, x):
        A = np.array([[-1, -2, 1],
                      [2, -1, 1],
                      [0, 0, -3]])
        return A @ x

    # Initial condition
    x0 = [1, 0, 1]
    t_span = (0, 10)

    # Solve the system
    sol = solve_ivp(system, t_span, x0, t_eval=np.linspace(0, 10, 100))

    # Plotting the results
    plt.plot(sol.t, sol.y[0], label="x1(t)")
    plt.plot(sol.t, sol.y[1], label="x2(t)")
    plt.plot(sol.t, sol.y[2], label="x3(t)")
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.legend()
    plt.show()

    return sol


def plot_phase_portraits(sol):
    # 2D projections
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("x1-x2 plane")
    plt.show()

    plt.plot(sol.y[0], sol.y[2])
    plt.xlabel("x1")
    plt.ylabel("x3")
    plt.title("x1-x3 plane")
    plt.show()

    plt.plot(sol.y[1], sol.y[2])
    plt.xlabel("x2")
    plt.ylabel("x3")
    plt.title("x2-x3 plane")
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.show()


def main():
    while True:
        print("\nSelect a task to execute:")
        print("1: Compute Eigenvalues and Eigenvectors")
        print("2: Determine Stability of the Equilibrium at x=0")
        print("3: Solve the Linear System and Plot Trajectories")
        print("4: Plot Phase Portraits")
        print("0: Exit")

        choice = input("Enter your choice (0-4): ")

        if choice == '1':
            compute_eigenvalues_eigenvectors()
        elif choice == '2':
            determine_stability()
        elif choice == '3':
            sol = solve_linear_system()
        elif choice == '4':
            # Ensure the system is solved first to plot the portraits
            try:
                sol
            except NameError:
                print("You need to solve the system first by selecting task 3.")
            else:
                plot_phase_portraits(sol)
        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a number between 0 and 4.")


if __name__ == "__main__":
    main()
