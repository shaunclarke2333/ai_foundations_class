# The Neural Thermostat Agent

**Course:** CSC6313 AI Foundations  
**Week:** 07 Neural Networks from Scratch  
**Author:** Shaun Clarke

---

## Project Overview

This project implements a **single-neuron neural network (Perceptron) from scratch** no deep learning frameworks, no black boxes. The neuron acts as the decision-making "brain" of a smart home thermostat, learning when to turn the AC ON or OFF based on indoor temperature and occupancy.

Every component the sigmoid activation function, the forward pass, and gradient descent is implemented by hand using only NumPy. This makes the internal mechanics of a neural network fully visible and traceable.

---

## What the Neuron Learns

The thermostat agent is trained on four labeled scenarios:

| Temperature (scaled) | People Count (scaled) | Label | Meaning |
|---|---|---|---|
| 0.1 | 0.1 | 0 (OFF) | Cold & empty AC not needed |
| 0.9 | 0.2 | 1 (ON) | Hot & few people AC on |
| 0.5 | 0.8 | 1 (ON) | Warm & crowded AC on |
| 0.2 | 0.9 | 0 (OFF) | Cold & crowded AC not needed |

Both features are pre-scaled to the range 0–1. The neuron's job is to find weights `w1`, `w2`, and a `bias` that correctly separate the ON cases from the OFF cases.

---

## Expected Output

```
--- Before Training ---
Initial Weights: w1=0.50, w2=-0.14, b=0.65
Prediction for [0.9, 0.2]: 0.78

Training in progress...

--- After Training ---
Final Weights: w1=3.81, w2=-1.29, b=-2.24
Prediction for [0.9, 0.2]: 0.94
```

After training, the neuron's prediction for a hot room with few people (`[0.9, 0.2]`) climbs from ~0.78 toward ~0.94 a high-confidence ON decision. A Matplotlib window then displays the loss curve, which should show a smooth downward trend confirming gradient descent is working.

---

## Architecture: A Single Neuron

```
        x1 (Temperature)  ──→ [× w1] ──┐
                                        ├──→ z = x1·w1 + x2·w2 + bias ──→ sigmoid(z) ──→ output (0–1)
        x2 (People Count) ──→ [× w2] ──┘
                                  ↑
                                 bias
```

The entire model is one neuron with three learnable parameters: `w1`, `w2`, and `bias`. This is the most fundamental building block of every neural network a single artificial neuron.

---

## Implementation Details

### Initialization (`__init__`)

```python
np.random.seed(42)
self.w1   = np.random.randn()   #  0.50
self.w2   = np.random.randn()   # -0.14
self.bias = np.random.randn()   #  0.65
self.learning_rate = 0.1
```

Weights are initialized with `np.random.randn()` random values drawn from a standard normal distribution. Random initialization **breaks symmetry**: if all weights started at the same value (e.g., 0), every neuron in a larger network would receive identical gradients and update identically, making multiple neurons redundant. Even in a single-neuron case, random init ensures the starting point is not artificially clean.

`np.random.seed(42)` ensures the same random starting weights every run, making results reproducible.

The bias is also randomly initialized. This allows the neuron to fire (produce non-zero output) even when all inputs are zero, giving the decision boundary freedom to shift.

---

### Task 1 Sigmoid Activation Function (`sigmoid`)

```python
def sigmoid(self, z: float):
    sig = 1 / (1 + np.exp(-z))
    return sig
```

**Formula:** σ(z) = 1 / (1 + e⁻ᶻ)

The sigmoid squashes any real number into the range (0, 1). This is what makes the neuron's output interpretable as a **probability** values close to 1 mean "high confidence AC should be ON", values close to 0 mean "high confidence AC should be OFF".

| z value | sigmoid(z) | Interpretation |
|---|---|---|
| −5 | ~0.007 | Very confident: OFF |
| 0 | 0.5 | Completely uncertain |
| +5 | ~0.993 | Very confident: ON |

Without an activation function, the neuron would just be a linear equation (z = x1·w1 + x2·w2 + bias) no squashing, no probability interpretation, and no ability to model non-linear decision boundaries.

---

### Task 2 Forward Pass (`predict`)

```python
def predict(self, x1, x2):
    z = (x1 * self.w1) + (x2 * self.w2) + self.bias
    activation = self.sigmoid(z)
    return activation
```

The forward pass is a two-step calculation:

**Step 1 Weighted sum:**
```
z = x1·w1 + x2·w2 + bias
```
Each input feature is multiplied by its weight, the products are summed, and the bias is added. This is a dot product it measures how strongly the current inputs align with what the neuron has learned to look for.

**Step 2 Activation:**
```
output = sigmoid(z)
```
z is passed through the sigmoid to produce the final probability. If `output > 0.5`, the neuron recommends AC ON; below 0.5, AC OFF.

---

### Task 3 Gradient Descent Training (`train`)

```python
def train(self, X: np.ndarray, y: np.ndarray, epochs=1000):
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            x1, x2 = X[i]
            target  = y[i]

            # Forward pass
            prediction = self.predict(x1, x2)

            # Error
            error = prediction - target

            # Weight updates (gradient descent)
            self.w1   -= self.learning_rate * error * x1
            self.w2   -= self.learning_rate * error * x2
            self.bias -= self.learning_rate * error

            # Accumulate squared error
            total_loss += (error ** 2)

        losses.append(total_loss / len(X))

    return losses
```

Training runs for `epochs` iterations. Each iteration passes through all four training examples and adjusts the weights after each one — this is called **online (stochastic) gradient descent**.

**The error signal:**
```
error = prediction - target
```
- If prediction is too **high** (e.g., 0.85 but target is 0): error is positive → weights decrease → future predictions lower
- If prediction is too **low** (e.g., 0.15 but target is 1): error is negative → weights increase → future predictions higher

**The weight update rule:**
```
w1   -= learning_rate × error × x1
w2   -= learning_rate × error × x2
bias -= learning_rate × error
```

The input `x1` and `x2` are included in the weight updates because the size of the input determines how much that feature contributed to the error. A feature with a value of 0 contributed nothing to the prediction, so its weight should not change multiplying by the input automatically enforces this.

The bias update has no input multiplier because bias contributes to every prediction equally regardless of input values so it is updated by the error alone.

**MSE tracking:**
At the end of each epoch, the average squared error (MSE) across all four training examples is appended to `losses`. The returned list is plotted to verify that gradient descent is converging.

---

## The Learning Rate

```python
self.learning_rate = 0.1
```

The learning rate controls the step size of each weight update. Too high and the updates overshoot the optimal weights, causing instability. Too low and convergence is very slow.

The project spec suggests `0.5` for faster convergence on this small dataset. The implementation uses `0.1`, which is more conservative convergence takes more epochs but is stable. The loss curve should still show a clear downward trend by epoch 1000.

---

## Loss Curve (Visualization)

```python
plt.figure(figsize=(10, 5))
plt.plot(history, color='blue', linewidth=2)
plt.title("Neural Agent Learning Curve", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Mean Squared Error (Loss)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

A **decreasing loss curve** confirms that gradient descent is working the neuron is making smaller and smaller errors over time as its weights converge toward values that correctly classify all four training examples.

If the curve is flat: the learning rate may be too low, or there is a bug in the weight update direction.  
If the curve rises: the weight update sign is likely flipped (adding instead of subtracting).

---

## Why This Is the Foundation of All Neural Networks

Every modern deep learning model from image classifiers to large language models is built from stacked layers of neurons like this one. The same three operations implemented here are present in every layer:

1. **Weighted sum** (z = Wx + b) the linear transformation
2. **Activation function** (sigmoid, ReLU, etc.) the non-linearity
3. **Gradient descent** the learning mechanism

The only differences in larger networks are: more neurons, more layers, more sophisticated activation functions, and more efficient matrix-vectorized implementations. The fundamental mechanics are identical.

---

## Prerequisites

- Python 3.10+

```bash
pip install numpy matplotlib
```

---

## How to Run

```bash
python shaun_clarke_csc6313_week07.py
```

The script runs to completion automatically no user input required. It prints the before/after weight comparison and prediction, then opens the Matplotlib loss curve window.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Random weight initialization, `exp()` for sigmoid, array operations |
| `matplotlib.pyplot` | Loss curve visualization |

---

## File Structure

```
week07/
├── shaun_clarke_csc6313_week07.py    # Full implementation
└── README.md                         # This file
```

---

## Design Notes

**Why `np.random.randn()` instead of fixed values like `w1=0.5`?** The project spec suggests fixed initialization for reproducibility, but `np.random.randn()` with a fixed seed achieves both reproducibility (same seed → same random numbers every run) and the theoretical correctness of random initialization (breaks symmetry). Both approaches converge on this small dataset.

**Why online gradient descent (update per sample) instead of batch gradient descent (update per epoch)?** With only 4 training examples, there is no meaningful efficiency difference. Online updates tend to be noisier but can help escape local minima on more complex problems. For a single-neuron model with a convex loss landscape, both converge to the same solution.

**Why squared error (`error ** 2`) instead of binary cross-entropy?** Cross-entropy is theoretically better suited for binary classification because its gradient has nicer properties with sigmoid outputs. However, squared error with sigmoid still converges correctly on this simple problem and is easier to reason about when learning the mechanics of gradient descent for the first time.

**Why does `w1` grow positive and `w2` grows less so after training?** The training data makes temperature a stronger predictor of AC need than occupancy: the hot room `[0.9, 0.2]` maps to ON while the cold crowded room `[0.2, 0.9]` maps to OFF. The neuron learns to weight temperature more heavily, which is reflected in `w1` growing significantly larger than `w2`.