# Gated Recurrent Unit (GRU)

## Introduction

The LSTM solved the vanishing gradient problem at the cost of architectural complexity: two state vectors, three gates, and four times the parameters of a vanilla RNN. A natural question arises: is all of that complexity necessary?

The **Gated Recurrent Unit**, introduced by Cho et al. (2014), answers with a streamlined alternative. The GRU merges the cell state and hidden state into a single vector, reduces to two gates, and achieves comparable performance to the LSTM on most tasks with approximately 25% fewer parameters. Understanding the GRU requires seeing it not merely as a simplified LSTM but as a different design philosophy — one that challenges the assumption that separate memory and output states are necessary.

---

## 1. Motivation: Simplifying the Gating Mechanism

### What the LSTM Separates

The LSTM maintains a principled separation:
- $c_t$: long-term memory (cell state), never directly exposed
- $h_t$: working memory (hidden state), the actual output

The forget gate, input gate, and output gate each serve distinct roles in mediating between these two states.

### The GRU's Hypothesis

The GRU proposes that this separation is redundant. A single gated hidden state, updated with two carefully designed gates, can capture the same essential dynamics with a simpler structure.

Empirically, this hypothesis has held up: across a wide range of tasks, GRUs match LSTMs in performance while being faster to train and easier to tune.

---

## 2. GRU Architecture

The GRU maintains a single state vector $h_t \in \mathbb{R}^H$ and uses two gates.

### Reset Gate

The **reset gate** $r_t$ controls how much of the previous hidden state is used when computing the candidate update:

$$r_t = \sigma\left(W_r \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_r\right)$$

- $r_t \approx 0$: ignore the previous hidden state entirely — compute the candidate from $x_t$ alone
- $r_t \approx 1$: use the full previous hidden state — same behavior as a vanilla RNN

### Update Gate

The **update gate** $z_t$ controls the interpolation between the previous hidden state and the candidate:

$$z_t = \sigma\left(W_z \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_z\right)$$

- $z_t \approx 0$: discard the candidate, retain the previous hidden state (skip the update)
- $z_t \approx 1$: replace the previous hidden state with the candidate (full update)

### Candidate Hidden State

The candidate $\tilde{h}_t$ is a proposed new value for the hidden state, computed using the reset-gated previous state:

$$\tilde{h}_t = \tanh\left(W_h \begin{bmatrix} r_t \odot h_{t-1} \\ x_t \end{bmatrix} + b_h\right)$$

The reset gate $r_t$ acts *inside* this computation, allowing the candidate to be computed as if the past were partially or fully forgotten.

### State Update

The final hidden state is a **linear interpolation** between the previous state and the candidate:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

This can be rewritten as:

$$h_t = h_{t-1} + z_t \odot (\tilde{h}_t - h_{t-1})$$

which makes the additive structure explicit: the update gate controls how much the hidden state moves toward the candidate.

### Output

$$\hat{y}_t = g\left(W_y h_t + b_y\right)$$

---

## 3. Structural Comparison with LSTM

| Component | LSTM | GRU |
|---|---|---|
| State vectors | $h_t$, $c_t$ | $h_t$ only |
| Number of gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Explicit, separate | Merged into $h_t$ |
| Candidate update | $i_t \odot \tilde{c}_t$ added to $c_t$ | $z_t$ interpolates $h_{t-1}$ and $\tilde{h}_t$ |
| Output exposure | $o_t \odot \tanh(c_t)$ | $h_t$ directly |
| Parameters | $4H(H + d + 1)$ | $3H(H + d + 1)$ |

### Functional Correspondences

The update gate $z_t$ in the GRU combines the roles of the LSTM's forget and input gates. Setting $z_t = 1 - f_t = i_t$ (with coupled forget and input) recovers a coupled-gate LSTM variant. The GRU makes this coupling the default.

The reset gate $r_t$ has no direct LSTM counterpart — it controls how much history influences the *candidate* computation itself, rather than modulating a gate applied to an already-computed candidate.

---

## 4. Gradient Flow Analysis

### Backpropagation Through the Update

The gradient of $h_t$ with respect to $h_{t-1}$ through the update equation:

$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

The term $(1 - z_t)$ provides a direct additive gradient path from $h_t$ back to $h_{t-1}$, bypassing nonlinear activations entirely.

Across $k$ time steps, this path contributes:

$$\frac{\partial h_t}{\partial h_{t-k}} \supseteq \prod_{j=t-k+1}^{t} (1 - z_j)$$

When $z_j \approx 0$ (update gate closed), the factor is near 1 — the GRU retains the previous hidden state and propagates gradients without decay, directly analogous to the LSTM's forget gate mechanism via the cell state.

### Key Difference from LSTM

In the LSTM, the gradient highway runs through $c_t$ and involves only $f_t$. In the GRU, it runs through $h_t$ directly and involves $(1 - z_t)$. The GRU's hidden state simultaneously serves as memory, gradient highway, and output — a design that works because the update gate can selectively freeze its own state.

---

## 5. Parameter Count

Each gate and the candidate involve a weight matrix of size $H \times (H + d)$ and a bias of size $H$. With three such transformations (reset gate, update gate, candidate):

$$|\theta_{\text{GRU}}| = 3 \cdot \left[H(H + d) + H\right] = 3H(H + d + 1)$$

Compared to LSTM with $4H(H + d + 1)$, the GRU uses **25% fewer parameters** for the same hidden size. In practice, this means:

- Faster training (fewer floating-point operations per step)
- Lower memory footprint
- Reduced overfitting risk on small datasets
- Slightly reduced representational capacity

---

## 6. Implementation in PyTorch

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.gru(x)
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) — last hidden state
        return self.fc(out[:, -1, :])   # prediction from last timestep

model = GRUModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
```

Note that unlike `nn.LSTM`, `nn.GRU` returns only a single state $h_n$ (no $c_n$), reflecting the merged state design.

### Direct Comparison in a Training Loop

```python
lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
gru_model  = GRUModel(input_size=1, hidden_size=64, num_layers=2)

# GRU trains faster — fewer parameters, same hidden_size
lstm_params = sum(p.numel() for p in lstm_model.parameters())
gru_params  = sum(p.numel() for p in gru_model.parameters())

print(f"LSTM parameters: {lstm_params:,}")   # ~66,049
print(f"GRU  parameters: {gru_params:,}")    # ~49,537  (~25% fewer)
```

### Key Hyperparameters

| Parameter | Typical Range | Effect |
|---|---|---|
| `hidden_size` | 32–512 | Capacity of the merged state $h_t$ |
| `num_layers` | 1–3 | Hierarchical temporal abstraction |
| `dropout` | 0.1–0.5 | Regularization between stacked layers |
| Lookback window | 30–200 | Temporal context provided as input |

---

## 7. Strengths and Weaknesses

### Strengths

**Parameter efficiency**: 25% fewer parameters than an LSTM at identical hidden size. For the same computational budget, the GRU can use a larger hidden state than the LSTM, potentially recovering or exceeding representational capacity despite the simpler gating.

**Faster training**: Fewer matrix multiplications per time step translate directly to faster training epochs. On long sequences or resource-constrained settings, this is a practical advantage.

**Competitive performance**: Across empirical comparisons on language modeling, machine translation, and time series forecasting, GRUs consistently match LSTM performance and occasionally exceed it on smaller datasets. The simpler architecture is less prone to overfitting when data is limited.

**Simpler gradient flow**: With a single state vector, the gradient dynamics are easier to analyze and debug. The update gate's direct interpolation structure makes the gradient highway more transparent than the LSTM's dual-state system.

**Fewer tuning decisions**: No separate cell state initialization, no forget gate bias initialization trick, no output gate to tune. The GRU's simpler design reduces the hyperparameter surface.

**Natural default for moderate-length sequences**: For financial time series, speech features, and similar tasks with sequences of 50–500 steps, the GRU is often the first gated architecture to reach acceptable performance during experimentation.

### Weaknesses

**Merged state reduces representational separation**: The LSTM's separation of $c_t$ (long-term memory) and $h_t$ (working output) is structurally motivated: different components of the network perform different roles. The GRU's merged state must serve both purposes simultaneously, which can be a limitation on tasks requiring very different timescales of memory.

**No output gate**: The LSTM's output gate $o_t$ controls which parts of the cell state are exposed as the hidden state, acting as a filter. The GRU exposes $h_t$ directly, which means the full (gated) hidden state is always the output. This is less flexible for tasks where memory and output have distinct structure.

**Performance gap on complex tasks**: On tasks with rich hierarchical temporal structure — long-horizon language modeling, complex sequence transduction — the LSTM's additional parameters and explicit memory separation tend to give a consistent, if modest, edge.

**Still sequential**: Like LSTM, the GRU cannot parallelize across time. Each $h_t$ depends on $h_{t-1}$, making training on very long sequences slower than attention-based models.

**Not universally simpler to interpret**: While the architecture is simpler than LSTM, the reset gate's action inside the candidate computation (rather than on an already-computed output) can make the information routing harder to trace than the LSTM's more modular gate structure.

---

## 8. GRU vs LSTM: When to Use Which

The empirical literature provides relatively consistent guidance:

**Prefer GRU when**:
- Training data is limited (lower parameter count reduces overfitting risk)
- Computational resources are constrained (faster training, lower memory)
- The task involves moderate sequence lengths (50–500 steps)
- You need a fast baseline before committing to a more complex architecture

**Prefer LSTM when**:
- The task requires modeling phenomena at very different timescales simultaneously
- Sequence lengths are very long and the additional long-term memory capacity helps
- You have sufficient data to support the increased parameter count
- The explicit cell state provides useful structure for downstream tasks (e.g., as an interpretable memory probe)

**In practice**: start with GRU, switch to LSTM if performance is insufficient after proper tuning. The difference is often within noise on typical financial time series datasets.

---

## 9. Extensions and Variants

### Minimal Gated Unit (MGU)

Further simplification using a single gate $f_t$ that simultaneously controls forgetting and updating:

$$\tilde{h}_t = \tanh(W_h (f_t \odot h_{t-1}) + W_x x_t + b)$$
$$h_t = (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t$$

Competitive on some tasks with ~33% fewer parameters than GRU.

### Coupled Input-Forget Gate LSTM

An LSTM variant that sets $i_t = 1 - f_t$, reducing the parameter count toward GRU territory while retaining the separate cell state:

$$c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t$$

### Bidirectional GRU

For tasks where future context is available (e.g., offline sequence labeling, not real-time forecasting), a bidirectional GRU processes the sequence in both directions and concatenates hidden states:

```python
bigru = nn.GRU(input_size=1, hidden_size=64, num_layers=2,
               batch_first=True, bidirectional=True)
# Output: (batch, seq_len, 2 * hidden_size)
```

This doubles the parameter count but provides richer context for each time step.

---

## 10. Summary

1. **The GRU merges the LSTM's cell state and hidden state** into a single vector $h_t$, controlled by two gates: the reset gate $r_t$ and the update gate $z_t$.

2. **The reset gate** determines how much of the previous hidden state influences the candidate update, enabling the model to compute new candidates as if the past were partially forgotten.

3. **The update gate** implements a linear interpolation between the previous state and the candidate — directly analogous to the LSTM's coupled forget-input mechanism, but applied to the unified hidden state.

4. **Gradient flow** is preserved through the $(1 - z_t)$ term in the update equation, providing an additive gradient highway when the update gate is closed.

5. **The GRU uses 25% fewer parameters than the LSTM** for the same hidden size, trains faster, and matches LSTM performance on most practical tasks. The LSTM retains a modest advantage on tasks requiring very long-range memory or explicit timescale separation.

6. **GRU is the recommended default** for new sequential modeling experiments when starting without strong prior knowledge — its lower complexity makes it faster to iterate on before committing to the LSTM's additional capacity.

---

## References

- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *NIPS Deep Learning Workshop*.
- Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A search space odyssey. *IEEE Transactions on Neural Networks and Learning Systems*.
- Zhou, G. B., Wu, J., Zhang, C. L., & Zhou, Z. H. (2016). Minimal gated unit for recurrent neural networks. *International Journal of Automation and Computing*.

---

**Previous Lecture**: Long Short-Term Memory (LSTM)
