# DSML Lab Assignment 2 - Problem 1: Linear System Analysis

## Problem Overview

We are analyzing a 3-dimensional linear continuous-time dynamical system given by the equation:

\[
\frac{dx}{dt} = A x
\]

where \( A \) is a matrix that defines how the system evolves over time, and \( x(t) \) is the state vector in \( \mathbb{R}^3 \).

### Tasks and Solution Reasoning

### Task 1: Compute the Eigenvalues and Eigenvectors of Matrix A

The first step in solving the system is to compute the eigenvalues and eigenvectors of the matrix \( A \). The eigenvalues give us important information about the nature of the solutions to the system. Specifically, they tell us whether the system is stable or unstable and whether the solutions grow, decay, or oscillate over time. 

The eigenvectors, on the other hand, define the directions in which the system evolves. In this case, the general solution to the system can be expressed as a combination of exponential terms involving the eigenvalues and eigenvectors. Therefore, computing these two elements is crucial for understanding the behavior of the system.

### Task 2: Determine the Stability of the Equilibrium Point at \( x = 0 \)

Once we have the eigenvalues, we can analyze the stability of the system at the equilibrium point \( x = 0 \). The stability of a system is determined by the real parts of its eigenvalues:
- If all the real parts are negative, the system is stable, meaning any perturbation will decay over time, and the system will return to equilibrium.
- If any eigenvalue has a positive real part, the system is unstable, and small disturbances will cause the system to move away from equilibrium.

Thus, by inspecting the eigenvalues, we can determine whether the system will naturally return to equilibrium or diverge from it over time.

### Task 3: Solve the System Using an ODE Solver

To understand how the system evolves over time from a specific initial condition, we numerically solve the system using an ODE solver. This allows us to simulate the behavior of the system over a given time period.

By solving the system for a specific initial condition, we can plot how the components of the state vector (i.e., \( x_1(t), x_2(t), x_3(t) \)) change over time. This provides a visual understanding of how the system behaves dynamically, whether it converges, diverges, or oscillates based on the nature of the eigenvalues.

### Task 4: Plot Phase Portraits

In addition to looking at the time evolution of the system, we can also plot phase portraits, which provide insights into how the components of the system relate to one another. Phase portraits allow us to visualize the trajectory of the system in different state spaces, such as the \( x_1 \)-\( x_2 \) plane or the \( x_1 \)-\( x_3 \) plane.

The phase portraits help us see patterns in the system’s behavior that may not be obvious from just looking at the time evolution. For example, it can show whether the system follows a stable spiral, oscillates, or moves towards or away from an equilibrium point.

### Conclusion

Through these tasks, we were able to analyze the dynamics of a linear system:
1. We used eigenvalues and eigenvectors to understand the fundamental structure of the system.
2. We assessed the system's stability by examining the real parts of the eigenvalues.
3. We simulated the system's behavior using numerical methods to visualize the time evolution.
4. Finally, we plotted phase portraits to gain additional insights into the system's overall dynamics.

In this particular case, the system was found to be unstable due to the presence of a positive eigenvalue, which indicates that disturbances from the equilibrium point will grow over time.
