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

 

class NeuralThermostat:

    def __init__(self):

        # Initializing weights and bias with random values

        np.random.seed(42)

        self.w1 = np.random.randn()

        self.w2 = np.random.randn()

        self.bias = np.random.randn()

        self.learning_rate = 0.1

      

    def sigmoid(self, z):

        # TASK 1: Implement the Sigmoid Activation Function

        # Formula: 1 / (1 + e^-z)

        # Hint: Use np.exp(-z) for the exponential part

        pass 

    

    def predict(self, x1, x2):

        # TASK 2: Implement the Feedforward Pass

        # 1. Calculate Z using the formula: (x1 * w1) + (x2 * w2) + b

        # 2. Pass Z through your sigmoid function and return the result

        pass

 

    def train(self, X, y, epochs=1000):

        losses = []

        for epoch in range(epochs):

            total_loss = 0

            for i in range(len(X)):

                x1, x2 = X[i]

                target = y[i]

                

                # --- TASK 3: THE TRAINING ENGINE ---

                # 1. Get the current prediction (Call your predict function)

                # prediction = ...

 

                # 2. Calculate the Error (The difference between prediction and target)

                # error = ...

                

                # 3. Update the weights and bias (Gradient Descent)

                # Formula: Weight = Weight - (Learning_Rate * Error * Input)

                # self.w1 = ...

                # self.w2 = ...

                # self.bias = ... (Hint: The 'input' for bias is always 1)

                

                # 4. Record the Squared Error for the loss chart

                # total_loss += (error ** 2)

                

                pass 

            

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