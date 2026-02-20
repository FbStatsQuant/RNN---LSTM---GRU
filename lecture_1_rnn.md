# Recurrent Neural Networks (RNNs)

## Introduction

Feedforward networks, as developed in previous lectures, assume that inputs are independent and identically distributed. This assumption fails for **sequential data**, where the order of observations carries information: time series, text, audio, and financial prices all exhibit temporal dependencies that a static architecture cannot capture.

**Recurrent Neural Networks** address this by introducing a hidden state that persists across time steps, allowing the network to accumulate information over a sequence. This lecture develops the mathematical structure of RNNs, their training procedure, and their fundamental limitations.

---

## 1. The Sequential Modeling Problem

### Setup

Consider a sequence of observations:

$$\mathbf{x} = (x_1, x_2, \ldots, x_T), \quad x_t \in \mathbb{R}^d$$

Our objective is to model temporal dependencies — either to predict a future value $y_{t+1}$ given $x_1, \ldots, x_t$ (autoregressive forecasting) or to assign a label to the entire sequence.

### Why Feedforward Networks Fail

A feedforward network $f_\theta : \mathbb{R}^d \to \mathbb{R}$ applied independently at each step treats $x_t$ in isolation. Even if we concatenate a fixed window $(x_{t-k}, \ldots, x_t)$, the model:

- Uses a fixed context window — cannot adapt to variable-length dependencies
- Has no memory — state at $t$ does not inform state at $t+1$
- Has no parameter sharing — separate weights for each position

The recurrent architecture solves all three problems by design.

---

## 2. RNN Architecture

### Hidden State Recurrence

The core idea is a **hidden state** $h_t \in \mathbb{R}^H$ that summarizes information up to time $t$. It is updated recursively:

$$h_t = \phi\left(W_h h_{t-1} + W_x x_t + b\right)$$

where:

- $W_h \in \mathbb{R}^{H \times H}$ is the **recurrent weight matrix**
- $W_x \in \mathbb{R}^{H \times d}$ is the **input weight matrix**
- $b \in \mathbb{R}^H$ is the **bias vector**
- $\phi$ is a nonlinear activation, typically $\tanh$
- $h_0 = \mathbf{0}$ (zero initialization)

At each step $t$, the output is:

$$\hat{y}_t = g\left(W_y h_t + b_y\right)$$

where $g$ is the task-appropriate output activation (identity for regression, softmax for classification).

### Parameter Sharing

A critical property: **the same parameters** $\{W_h, W_x, b, W_y, b_y\}$ are applied at every time step. This gives the RNN:

- **Translation invariance in time**: the same pattern is recognized regardless of when it occurs
- **Efficient parameterization**: the number of parameters is $O(H^2 + Hd)$, independent of sequence length $T$

### Computational Graph

The unrolled computation across $T$ steps forms a directed acyclic graph:

$$x_1 \to h_1 \to h_2 \to \cdots \to h_T \to \hat{y}$$
$$\quad\quad\quad\searrow\quad\quad\searrow$$
$$\quad\quad\quad \hat{y}_1 \quad\quad \hat{y}_2$$

Each $h_t$ depends on both the current input $x_t$ and the previous state $h_{t-1}$, creating the chain of temporal dependencies.

---

## 3. Training: Backpropagation Through Time (BPTT)

### Loss Function

For a sequence prediction task, the total loss sums contributions across time steps:

$$\mathcal{L}(\theta) = \sum_{t=1}^T \ell\left(y_t, \hat{y}_t\right)$$

For regression with MSE: $\ell(y_t, \hat{y}_t) = (y_t - \hat{y}_t)^2$

### The BPTT Algorithm

Standard backpropagation applied to the unrolled computational graph. The gradient with respect to the recurrent weights at time $t$ requires differentiating through the entire chain of hidden states back to $t=1$.

By the chain rule, the gradient of the loss at step $t$ with respect to $h_k$ for $k \leq t$ is:

$$\frac{\partial \mathcal{L}_t}{\partial h_k} = \frac{\partial \mathcal{L}_t}{\partial h_t} \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$$

Each factor in the product is:

$$\frac{\partial h_j}{\partial h_{j-1}} = W_h^T \cdot \text{diag}\left(\phi'(z_j)\right)$$

where $z_j = W_h h_{j-1} + W_x x_j + b$.

The total gradient of the recurrent weight matrix accumulates contributions from all time steps:

$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^T \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_t} \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_h}$$

### Truncated BPTT

For long sequences, full BPTT becomes computationally expensive and numerically unstable. **Truncated BPTT** limits the backward pass to a fixed window of $k$ steps:

$$\frac{\partial \mathcal{L}_t}{\partial h_{t-k}} \approx \frac{\partial \mathcal{L}_t}{\partial h_t} \prod_{j=t-k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$$

This is the standard approach in practice, trading off gradient accuracy for computational tractability.

---

## 4. The Vanishing and Exploding Gradient Problem

### Mathematical Origin

The product structure of BPTT is the source of the fundamental instability of vanilla RNNs. For a simplified scalar analysis, consider:

$$\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \approx (w_h)^{t-k} \cdot \prod_{j=k+1}^{t} \phi'(z_j)$$

Two regimes emerge as $t - k \to \infty$:

**Vanishing gradients** ($|w_h| < 1$ or $|\phi'| < 1$ consistently):

$$\left\|\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right\| \to 0 \quad \text{exponentially fast}$$

Gradients from distant time steps vanish, making it impossible to learn long-range dependencies.

**Exploding gradients** ($|w_h| > 1$):

$$\left\|\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right\| \to \infty$$

Gradients from distant time steps blow up, destabilizing training.

### Tanh and Saturation

With $\tanh$ activation: $|\phi'(z)| = 1 - \tanh^2(z) \leq 1$, with equality only at $z = 0$. In practice, activations saturate and $\phi'(z) \ll 1$, causing **vanishing gradients** to dominate.

This means vanilla RNNs have **no reliable mechanism to carry information across more than ~10–20 time steps**, regardless of architectural depth.

### Partial Mitigation

- **Gradient clipping**: $g \leftarrow g \cdot \min\left(1, \frac{\tau}{\|g\|}\right)$ addresses explosions but not vanishing
- **Careful initialization**: $W_h$ initialized close to orthogonal to preserve gradient norms
- **Truncated BPTT**: Limits the propagation distance, avoiding the worst instability

These are patches, not solutions. The fundamental fix requires architectural changes — which motivates LSTM and GRU.

---

## 5. Implementation in PyTorch

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input shape: (batch, seq_len, input_size)
            nonlinearity='tanh',       # or 'relu'
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.rnn(x)
        # out: (batch, seq_len, hidden_size)
        return self.fc(out[:, -1, :])  # use last timestep

model = RNNModel(input_size=1, hidden_size=64, num_layers=2)

# Gradient clipping during training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = criterion(model(X_batch), y_batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Key Hyperparameters

| Parameter | Typical Range | Effect |
|---|---|---|
| `hidden_size` | 32–512 | Capacity of the hidden state |
| `num_layers` | 1–3 | Depth of stacked RNN |
| `dropout` | 0.1–0.5 | Regularization between layers |
| `seq_len` (lookback) | 20–200 | Temporal context window |

---

## 6. Strengths and Weaknesses

### Strengths

**Parameter efficiency**: Shared weights across time steps give the RNN an inductive bias suited to sequential data. For short sequences (fewer than ~20 steps), this efficiency is a genuine advantage over transformers.

**Simplicity**: The architecture is conceptually minimal — a single recurrence equation. Fewer design choices, fewer failure modes, and easier to debug.

**Autoregressive generation**: The recurrent structure is naturally suited to generating sequences one step at a time, making RNNs intuitive for forecasting.

**Low latency inference**: Processing each new input requires only a single forward step through the recurrent cell, making RNNs efficient at test time for streaming applications.

### Weaknesses

**Vanishing gradients**: The defining limitation. Long-range dependencies — patterns that span tens or hundreds of steps — are effectively inaccessible to vanilla RNNs. This is not a tuning problem; it is structural.

**Sequential computation**: Hidden state $h_t$ depends on $h_{t-1}$, so the forward pass cannot be parallelized across time. Training on long sequences is slow compared to attention-based models.

**Memory bottleneck**: All history must be compressed into the fixed-size vector $h_t \in \mathbb{R}^H$. Long sequences with rich structure exceed the representational capacity of this compressed state.

**Sensitivity to sequence length**: Performance degrades predictably with sequence length due to gradient issues and the finite hidden state. This degradation is smoother in LSTM/GRU but sharper in vanilla RNNs.

**Superseded for most tasks**: For text and most long-sequence tasks, Transformers have replaced RNNs entirely. For financial time series with moderate lookback windows, LSTM and GRU are standard choices over vanilla RNNs.

---

## 7. When to Use Vanilla RNNs

Despite their limitations, vanilla RNNs remain appropriate in specific scenarios:

- **Short sequences** ($T \lesssim 20$) where long-range dependencies are absent
- **Resource-constrained environments** where LSTM/GRU gate computations are too expensive
- **Baseline comparison**: Always train a vanilla RNN before moving to gated architectures — it establishes a lower bound and confirms that added complexity is justified by the data

For sequences longer than ~20 steps with non-trivial temporal structure, the LSTM or GRU should be the default choice.

---

## 8. Summary

1. **RNNs model sequential data** by maintaining a hidden state $h_t$ that is updated at each time step using both the current input and the previous state, with shared parameters across all steps.

2. **Training uses BPTT**, which backpropagates gradients through the unrolled computational graph. Truncated BPTT limits computational cost but introduces approximation.

3. **The vanishing gradient problem** is the fundamental limitation: products of Jacobians shrink exponentially with sequence length, preventing learning of long-range dependencies. Gradient clipping addresses explosions but not vanishing.

4. **RNNs are parameter-efficient and fast at inference** but suffer from sequential computation, a fixed-size memory bottleneck, and structural inability to capture long-range patterns reliably.

5. For most practical applications involving sequences longer than ~20 steps, **gated architectures (LSTM, GRU)** are preferred — they were designed specifically to address the vanishing gradient problem.

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
- Werbos, P. J. (1990). Backpropagation through time: What it does and how to do it. *Proceedings of the IEEE*.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.

---

**Next Lecture**: Long Short-Term Memory (LSTM)
