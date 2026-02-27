"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Module: 7
Assignment: The Neural Thermostat Agent 

Objective:
You are building the “brain” of a Smart Home Thermostat. The agent’s goal is to decide whether to turn the Air Conditioning ON (1) or OFF (0) based on two environmental factors:
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


 

class NeuralThermostat:

    def __init__(self):

        # Initializing weights and bias with random values

        # The seed number specifies where to start generating random numbers from. This ensures the same numbers are generated all the time every time
        np.random.seed(42)

        # Initializing random weights to break the symmetry, meaning; no two weights will have the same value that will result in identical behavior during forward and backward passes
        self.w1: float = np.random.randn()
        self.w2: float = np.random.randn()
        # Initializing a random number for bias as well. This allows the neuron to still activate(nudging its decision boundary up or down) even if all inputs are zero.
        # If all inputs are 0, the weighted sum(input * weights) would be 0.
        self.bias: float = np.random.randn()
        # Setting the learning rate 0.1. The learning rate determines the rate at whcich the network learns. meaning how big is the amount by whihc each weight is updated after a mistake. 
        # Finding a sweet spot is good, a very low rate might be stable but makes training slow. Too high makes training faster, but also risks instability, becasue the icreases are to big and misses the optimal setting for the weights.
        self.learning_rate: float = 0.1

    def sigmoid(self, z: float):

        # TASK 1: Implement the Sigmoid Activation Function
        # Implementing the sigmoid function
        sig: float = 1 / (1 + np.exp(-z))
        
        return sig 

    def predict(self, x1, x2):

        # TASK 2: Implement the Feedforward Pass
        # Calculating the z which is the weighted sums plus the bias
        z: float = (x1 * self.w1) + (x2 * self.w2) + self.bias

        # passing z through the sigmoid (activation) function to get the neuron output
        activation: float = self.sigmoid(z)

        return activation

    def train(self, X: np.ndarray, y: np.ndarray, epochs=1000):

        losses: List = []

        for epoch in range(epochs):

            total_loss: float = 0

            for i in range(len(X)):
                # Initializing variables for features in the np array input
                x1: float
                x2: float

                # Unpacking the features in the np array
                x1, x2 = X[i]
                # Initialing the target variable
                target: np.ndarray = y[i]

                # --- TASK 3: THE TRAINING ENGINE ---
                # 1. Get the current prediction (Call your predict function)
                prediction = self.predict(x1, x2)

                # 2. Calculate the Error (The difference between prediction and target)
                # The error tells us which the direction the prediction is going based on the the size of the prediciton.
                # If the prediction is too Small, prediction = positive error. If the prediction is too large, prediction = negative error.
                error = prediction - target

                # 3. Update the weights and bias (Gradient Descent)
                # Formula: Weight = Weight - (Learning_Rate * Error * Input)
                # Using a very simplified case specific version of the gradient descent formula to update the weights for input 1.
                # The input is included because the size of the input allows the formula to uderstand just how much it needs to adjust the weights by. 
                self.w1 = self.w1 - (self.learning_rate * error * x1)
                # Using a very simplified case specific version of the gradient descent formula to update the weights for input 2
                # The input is included because the size of the input allows the formula to uderstand just how much it needs to adjust the weights by. 
                self.w2 = self.w2 - (self.learning_rate * error * x2)
                # Updating bias. We multiply the error by learning rate, because this lets us know by how much we need to adjust the bias by. subtracting ensures we move opposite teh slope (descent).
                # It's like standing on a hill, if the slope is positive and we subtract, we move left
                #  if the slope is negative and we subtract, we move right
                self.bias = self.bias - (self.learning_rate * error)

                # 4. Record the Squared Error for the loss chart
                total_loss += (error ** 2)

            # Keep track of the average loss for this epoch
            losses.append(total_loss / len(X))

        return losses

 

# --- PROJECT DATASET ---

# Features: [Temperature (0-1), PeopleCount (0-1)]

# Targets: 1 (AC On), 0 (AC Off)

X_train = np.array([

    [0.1, 0.1], # Cold & Empty -> Off (0)

    [0.9, 0.2], # Hot & Few people -> On (1)

    [0.5, 0.8], # Warm & Crowded -> On (1)

    [0.2, 0.9]  # Cold & Crowded -> Off (0)

])

y_train = np.array([0, 1, 1, 0])

 

# --- EXECUTION ---

agent = NeuralThermostat()

 

print("--- Before Training ---")

print(f"Initial Weights: w1={agent.w1:.2f}, w2={agent.w2:.2f}, b={agent.bias:.2f}")

# Test prediction for [Hot, Few People]

initial_pred = agent.predict(0.9, 0.2)

print(f"Prediction for [0.9, 0.2]: {initial_pred if initial_pred is not None else 'No Output Yet'}")

 

# Train the agent

print("\nTraining in progress...")

history = agent.train(X_train, y_train)

 

print("\n--- After Training ---")

print(f"Final Weights: w1={agent.w1:.2f}, w2={agent.w2:.2f}, b={agent.bias:.2f}")

final_pred = agent.predict(0.9, 0.2)

print(f"Prediction for [0.9, 0.2]: {final_pred if final_pred is not None else 'No Output Yet'}")

 

# --- VISUALIZATION ---

if history:

    plt.figure(figsize=(10, 5))

    plt.plot(history, color='blue', linewidth=2)

    plt.title("Neural Agent Learning Curve", fontsize=14)

    plt.xlabel("Epochs", fontsize=12)

    plt.ylabel("Mean Squared Error (Loss)", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()