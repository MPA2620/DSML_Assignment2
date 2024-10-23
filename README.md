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


# DSML Lab Assignment 2 - Problem 2: Two-Dimensional Hopfield Neural Network

## Problem Overview

We are tasked with analyzing a two-neuron continuous-time Hopfield network, described by the following differential equation:

\[
\frac{dx}{dt} = -x + W\sigma(x) + I
\]

Where:
- \( x(t) = \begin{bmatrix} x_1(t) \\ x_2(t) \end{bmatrix} \in \mathbb{R}^2 \) is the state vector.
- \( W = \begin{bmatrix} 0 & w_{12} \\ w_{21} & 0 \end{bmatrix} \) is the weight matrix, with synaptic weights \( w_{12} = w_{21} = w \).
- \( \sigma(x) = \tanh(x) \) is the activation function applied element-wise to \( x \).
- \( I = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \) is the constant external input vector in this problem.

### Task 1: Analyze the Stability of the Null Equilibrium Point

The first task is to analyze the stability of the trivial equilibrium point \( (x_1, x_2) = (0, 0) \). To solve this, we need to:

1. **Set up the Jacobian Matrix**:
   The Jacobian matrix of the system at \( (0, 0) \) tells us how small perturbations evolve around this equilibrium point. The matrix describes the local linear behavior of the system near the equilibrium.
   
   Since we are given that \( w_{12} = w_{21} = w \), the Jacobian at \( (0, 0) \) becomes:
   
   \[
   J = \begin{bmatrix} -1 & w \\ w & -1 \end{bmatrix}
   \]

2. **Eigenvalue Calculation**:
   The stability of the equilibrium point can be determined by analyzing the eigenvalues of the Jacobian. If the real part of all eigenvalues is negative, the equilibrium is stable. If any eigenvalue has a positive real part, the system is unstable at \( (0, 0) \).
   
   By solving for the eigenvalues of the Jacobian, we can determine how the value of \( w \) affects the stability of the system.

3. **Effect of \( w \)**:
   The parameter \( w \) affects the coupling between the two neurons. As we vary \( w \), the eigenvalues change, leading to different stability conditions at \( (0, 0) \). The analysis shows that when \( w \) increases, the system becomes more likely to be unstable.

### Task 2: Find Equilibrium Points for Different Values of \( w \)

In this task, we need to find and analyze the equilibrium points for three values of \( w \): \( w = 0.5, 1, 2 \).

1. **Trivial Equilibrium Point**:
   The trivial equilibrium point is always \( (x_1, x_2) = (0, 0) \). For each value of \( w \), we can determine the stability of this point by computing the Jacobian matrix as described in Task 1.

2. **Numerical Methods**:
   Although the problem suggests that the trivial equilibrium point may be the only solution for the given values of \( w \), for more complex systems, we would typically use numerical solvers to check for any non-trivial equilibrium points.

3. **Jacobian and Stability**:
   For each equilibrium point found, we calculate the Jacobian matrix and its eigenvalues to assess stability. This provides insights into whether the system will converge or diverge from these points.

### Task 3: Simulate the Hopfield Network

The final task is to simulate the Hopfield network for each value of \( w \) and visualize the system’s behavior.

1. **Implement the System of Differential Equations**:
   The system is implemented in Python using the differential equation \( \frac{dx}{dt} = -x + W\sigma(x) \). This describes how the states \( x_1 \) and \( x_2 \) evolve over time based on their interactions and the synaptic weights \( w \).

2. **Initial Conditions**:
   To explore different regions of the state space, we simulate the system for various initial conditions \( x(0) \). This helps visualize how the system behaves from different starting points.

3. **Numerical Simulation**:
   We solve the system of differential equations using a numerical ODE solver, such as `solve_ivp` from `scipy`. The system is simulated from \( t = 0 \) to \( t = 20 \), and the time evolution of \( x_1(t) \) and \( x_2(t) \) is plotted.

4. **Plotting the Direction Field**:
   In addition to plotting the trajectories, we visualize the direction field of the system in the \( x_1 - x_2 \) plane. This helps us see the general behavior of the system and how the trajectories flow through the state space.

5. **Overlaying the Trajectories**:
   Finally, we overlay the simulated trajectories on the direction field. This gives us a comprehensive view of how the system evolves over time and how it converges to or diverges from equilibrium points.

### Conclusion

This exercise allowed us to explore the dynamics of a simple two-neuron Hopfield network. We analyzed the stability of the equilibrium point \( (0, 0) \) for different values of \( w \), found equilibrium points, and simulated the system to visualize its behavior. Through this process, we gained insights into how the synaptic weights \( w \) affect the stability and overall dynamics of the network.
