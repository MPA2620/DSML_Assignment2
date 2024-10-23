import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torchdiffeq import odeint


# Van der Pol system of equations
def van_der_pol(t, z):
    x, y = z
    mu = 1.0  # Given in the problem
    dxdt = y
    dydt = mu * (1 - x ** 2) * y - x
    return [dxdt, dydt]


# Simulate the Van der Pol oscillator
def simulate_vdp():
    t_eval = np.linspace(0, 20, 1000)
    initial_conditions = [[2, 0], [0.5, 0.5], [-1, 2], [1, 0], [0, 0]]
    trajectories = []

    # Simulating the system for multiple initial conditions
    for ic in initial_conditions:
        sol = solve_ivp(van_der_pol, [0, 20], ic, t_eval=t_eval)
        trajectories.append((sol.t, sol.y))

    # Plot the trajectories
    for t, sol in trajectories:
        plt.plot(sol[0], sol[1], label=f"Initial: {ic}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Van der Pol Oscillator Trajectories')
    plt.legend()
    plt.show()

    return trajectories


# Define the neural network f(x; theta)
class NeuralODEFunc(nn.Module):
    def __init__(self):
        super(NeuralODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, t, x):
        return self.net(x)


# Define the neural ODE model
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.func = NeuralODEFunc()

    def forward(self, x0, t):
        return odeint(self.func, x0, t)


def train_neural_ode(train_data):
    # Convert training data to PyTorch tensors
    train_x = torch.tensor(np.concatenate([sol.T for _, sol in train_data]), dtype=torch.float32)

    # Ensure the time vector has the same number of steps as train_x
    num_time_steps = train_x.shape[0]  # This should match the number of rows in train_x
    train_t = torch.linspace(0, 10, num_time_steps)  # Adjust the end time (10) as needed

    # Initialize the model
    model = NeuralODE()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_x = model(train_x[0], train_t)  # Call to odeint happens here
        loss = criterion(pred_x, train_x)  # Shapes should match now
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


# Evaluate and visualize the model
def evaluate_model(model, test_data):
    # Convert test data to PyTorch tensors
    test_x = torch.tensor(np.concatenate([sol.T for _, sol in test_data]), dtype=torch.float32)

    # Ensure the time vector has the same number of steps as test_x
    num_time_steps = test_x.shape[0]  # Match the number of rows in test_x
    test_t = torch.linspace(0, 10, num_time_steps)  # Generate test_t with the same number of steps

    # Predict the test data
    with torch.no_grad():
        pred_test_x = model(test_x[0], test_t)

    # Compute test loss
    criterion = nn.MSELoss()
    test_loss = criterion(pred_test_x, test_x)
    print(f"Test Loss: {test_loss.item()}")

    # Plot the true vs predicted trajectories
    plt.plot(test_t, test_x[:, 0], label="True Trajectories (x)")
    plt.plot(test_t, pred_test_x[:, 0], '--', label="Predicted Trajectories (x)")
    plt.legend()
    plt.title("True vs Predicted Trajectories")
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.show()




# Main execution with menu
def main():
    while True:
        print("\nSelect a task to execute:")
        print("1: Simulate the Van der Pol Oscillator")
        print("2: Train Neural ODE model")
        print("3: Evaluate Neural ODE model")
        print("0: Exit")

        choice = input("Enter your choice (0-3): ")

        if choice == '1':
            # Task 1: Simulate the Van der Pol Oscillator
            trajectories = simulate_vdp()

            # Split data into training and testing sets (70% train, 30% test)
            global train_data, test_data
            train_data, test_data = [], []
            for t, sol in trajectories:
                split_idx = int(0.7 * len(t))
                train_data.append((t[:split_idx], sol[:, :split_idx]))
                test_data.append((t[split_idx:], sol[:, split_idx:]))

        elif choice == '2':
            # Task 2: Train Neural ODE model
            if 'train_data' not in globals():
                print("Please simulate the Van der Pol oscillator first (Task 1).")
            else:
                global model
                model = train_neural_ode(train_data)

        elif choice == '3':
            # Task 3: Evaluate Neural ODE model
            if 'model' not in globals():
                print("Please train the Neural ODE model first (Task 2).")
            else:
                evaluate_model(model, test_data)

        elif choice == '0':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please enter a number between 0 and 3.")


# Run the program
if __name__ == "__main__":
    main()
