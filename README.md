# Zero-Torch: Neural Network implementation FROM SCRATCH using only numpy (No Tensorflow/PyTorch)

This neural network is designed to classify handwritten digits from the MNIST dataset using only numpy. It uses:

* **Input layer**: 784 units (28×28 pixels)
* **Hidden layer**: 10 units with ReLU activation
* **Output layer**: 10 units with Softmax activation

---

## Forward Propagation

Given input $A^{[0]} = X \in \mathbb{R}^{784 \times m}$:

$$
\begin{aligned}
Z^{[1]} &= W^{[1]}A^{[0]} + b^{[1]} \\
A^{[1]} &= \text{ReLU}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]}A^{[1]} + b^{[2]} \\
A^{[2]} &= \text{softmax}(Z^{[2]})
\end{aligned}
$$

---

## Backward Propagation

$$
\begin{aligned}
dZ^{[2]} &= A^{[2]} - Y \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]T} \\
db^{[2]} &= \frac{1}{m} \sum dZ^{[2]} \\
dZ^{[1]} &= W^{[2]T} dZ^{[2]} \circ \text{ReLU}'(Z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} A^{[0]T} \\
db^{[1]} &= \frac{1}{m} \sum dZ^{[1]}
\end{aligned}
$$

---

## Parameter Updates

Using learning rate $\alpha$:

$$
\begin{aligned}
W^{[2]} &:= W^{[2]} - \alpha dW^{[2]} \\
b^{[2]} &:= b^{[2]} - \alpha db^{[2]} \\
W^{[1]} &:= W^{[1]} - \alpha dW^{[1]} \\
b^{[1]} &:= b^{[1]} - \alpha db^{[1]}
\end{aligned}
$$

---

## Shapes of Variables

### Forward Propagation

| Variable           | Shape    |
| ------------------ | -------- |
| $A^{[0]} = X$      | 784 × m  |
| $Z^{[1]}, A^{[1]}$ | 10 × m   |
| $W^{[1]}$          | 10 × 784 |
| $b^{[1]}$          | 10 × 1   |
| $Z^{[2]}, A^{[2]}$ | 10 × m   |
| $W^{[2]}$          | 10 × 10  |
| $b^{[2]}$          | 10 × 1   |

### Backward Propagation

| Gradient   | Shape    |
| ---------- | -------- |
| $dZ^{[2]}$ | 10 × m   |
| $dW^{[2]}$ | 10 × 10  |
| $db^{[2]}$ | 10 × 1   |
| $dZ^{[1]}$ | 10 × m   |
| $dW^{[1]}$ | 10 × 784 |
| $db^{[1]}$ | 10 × 1   |
