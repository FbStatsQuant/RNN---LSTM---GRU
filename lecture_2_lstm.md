# Long Short-Term Memory (LSTM)

## Introduction

The vanilla RNN's failure to capture long-range dependencies is not a hyperparameter problem — it is a structural one. The exponential decay of gradients through time means that no amount of tuning will enable reliable learning of patterns that span more than a few dozen steps.

The **Long Short-Term Memory** architecture, introduced by Hochreiter and Schmidhuber (1997), resolves this with a fundamentally different design: a **cell state** that carries information across time steps via additive updates, protected by learned gating mechanisms that control what information is written, retained, and read. This lecture develops the LSTM equations, explains the intuition behind each gate, and examines when and why this architecture succeeds where vanilla RNNs fail.

---

## 1. The Core Insight: Additive State Updates

### Why Additive Updates Help

Recall the vanishing gradient problem: products of Jacobians $\prod_{j} \frac{\partial h_j}{\partial h_{j-1}}$ shrink to zero exponentially because each factor involves $W_h^T \cdot \text{diag}(\phi'(z_j))$, and activation derivatives are bounded well below 1 in saturation.

The LSTM's solution is to replace the multiplicative recurrence with an **additive update** for the cell state $c_t$:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The gradient of $c_t$ with respect to $c_{t-1}$ is simply $f_t$ (the forget gate). When $f_t \approx 1$, this gradient is near 1 and does not decay — information and gradients can flow across arbitrarily many steps without vanishing.

This is the **Constant Error Carousel** mechanism: the cell state provides a direct, nearly gradient-free highway for information to travel across time.

---

## 2. LSTM Architecture

The LSTM maintains two state vectors at each time step:

- $h_t \in \mathbb{R}^H$: the **hidden state** (output state), same role as in the vanilla RNN
- $c_t \in \mathbb{R}^H$: the **cell state** (memory state), the long-term memory carrier

Both are initialized to zero: $h_0 = c_0 = \mathbf{0}$.

### Gates

Three learned gating vectors, each in $[0, 1]^H$ due to sigmoid activation, modulate information flow:

**Forget gate** — what fraction of the previous cell state to discard:

$$f_t = \sigma\left(W_f \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_f\right)$$

**Input gate** — what new information to write into the cell state:

$$i_t = \sigma\left(W_i \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_i\right)$$

**Output gate** — what portion of the cell state to expose as the hidden state:

$$o_t = \sigma\left(W_o \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_o\right)$$

### Candidate Cell State

A new candidate memory vector is computed using $\tanh$, bounded to $(-1, 1)^H$:

$$\tilde{c}_t = \tanh\left(W_c \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_c\right)$$

### State Updates

The cell state is updated additively:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The hidden state is derived from the cell state:

$$h_t = o_t \odot \tanh(c_t)$$

### Output

For regression or classification:

$$\hat{y}_t = g\left(W_y h_t + b_y\right)$$

where $g$ is the task-appropriate output activation.

---

## 3. Gate Semantics and Intuition

### Forget Gate $f_t$

$$f_t = \sigma(\cdot) \in (0, 1)^H$$

- $f_t \approx 0$: erase the cell state component (forget)
- $f_t \approx 1$: preserve the cell state component (remember)

The forget gate allows the model to **reset its memory** at the appropriate time. For example, when processing a financial time series, the model can learn to forget pre-event context after a structural break.

### Input Gate $i_t$ and Candidate $\tilde{c}_t$

These two work jointly:

- $\tilde{c}_t$: the new candidate content — what *could* be written
- $i_t$: the gating mask — what *actually gets* written

Their product $i_t \odot \tilde{c}_t$ is the actual update added to the cell state. This decoupling of *what* (candidate) from *how much* (gate) provides fine-grained control.

### Output Gate $o_t$

Controls how much of the cell state is exposed as the hidden state $h_t$. The $\tanh$ squashes $c_t$ to $(-1, 1)$ before gating, ensuring bounded output regardless of how large $c_t$ becomes from repeated additions.

### Joint Operation

The three gates work together to implement a **learned memory management system**:

1. Decide what to forget from the past ($f_t$)
2. Decide what new information to store ($i_t \odot \tilde{c}_t$)
3. Update the cell state additively ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$)
4. Decide what to output from the current memory ($o_t \odot \tanh(c_t)$)

---

## 4. Gradient Flow Through the Cell State

### Backpropagation Through the Cell State

The gradient of the loss with respect to $c_{t-1}$, flowing through $c_t$:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Across $k$ time steps:

$$\frac{\partial c_t}{\partial c_{t-k}} = \prod_{j=t-k+1}^{t} f_j$$

Since $f_j \in (0, 1)^H$, this product can still vanish if the forget gate consistently outputs values near zero. However:

- The network **learns** to set $f_t \approx 1$ when long-range memory is needed
- The gradient does not pass through $\tanh$ derivatives along this path — only through the forget gate values
- Vanishing is now **a choice the network makes**, not an inescapable structural property

This is qualitatively different from the vanilla RNN: gradients can flow arbitrarily far back when the forget gate is open.

### Gradient Through Hidden State

Gradients also flow through $h_t$ via the output gate and $\tanh(c_t)$. This path can still vanish, but the cell state path provides an alternative route that the optimizer can exploit.

---

## 5. Parameter Count

Each gate and the candidate state involve a weight matrix mapping $[h_{t-1}; x_t] \in \mathbb{R}^{H+d}$ to $\mathbb{R}^H$. With four such transformations:

$$|\theta_{\text{LSTM}}| = 4 \cdot \left[H(H + d) + H\right] = 4H(H + d + 1)$$

Compared to the vanilla RNN:

$$|\theta_{\text{RNN}}| = H(H + d + 1)$$

The LSTM uses **4× more parameters** than the vanilla RNN for the same hidden size. This is the direct cost of the gating mechanism.

---

## 6. Implementation in PyTorch

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) — last hidden state
        # c_n: (num_layers, batch, hidden_size) — last cell state
        return self.fc(out[:, -1, :])   # prediction from last timestep

model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
```

### Stacked LSTMs

Multiple LSTM layers can be stacked: the output sequence $h_1, \ldots, h_T$ of layer $\ell$ becomes the input sequence to layer $\ell+1$. PyTorch's `num_layers` parameter handles this automatically.

```python
# Two stacked LSTMs with dropout between layers
lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
               batch_first=True, dropout=0.3)
```

Dropout is applied between layers (not within a single layer's recurrence), preserving the temporal structure of gradients.

### Key Hyperparameters

| Parameter | Typical Range | Effect |
|---|---|---|
| `hidden_size` | 32–512 | Capacity of $h_t$ and $c_t$ |
| `num_layers` | 1–3 | Hierarchical temporal abstraction |
| `dropout` | 0.1–0.5 | Regularization between stacked layers |
| Lookback window | 30–200 | Temporal context provided as input |

---

## 7. Strengths and Weaknesses

### Strengths

**Long-range dependency learning**: The cell state with additive updates and learnable forget gates is the defining strength. LSTMs reliably learn dependencies spanning hundreds of steps — tasks completely inaccessible to vanilla RNNs.

**Selective memory**: The gating mechanism allows the model to make explicit learned decisions about what to store, discard, and expose at each step. This is a qualitative improvement over the single compressed hidden state of the RNN.

**Gradient stability**: By providing a near-constant gradient highway through the cell state, LSTMs are substantially more stable to train on long sequences. Vanishing gradients are a tunable failure mode, not an inherent one.

**Empirical track record**: LSTMs have dominated sequential modeling benchmarks across NLP, speech, and financial forecasting for two decades. There is an extensive literature on architecture variants, initialization strategies, and regularization techniques.

**Flexibility**: LSTMs can be applied as sequence-to-sequence, sequence-to-one, or one-to-sequence models with minor architectural modifications, making them versatile across tasks.

### Weaknesses

**Parameter overhead**: 4× the parameters of a vanilla RNN for the same hidden size. This increases memory requirements, training time, and overfitting risk on small datasets.

**Sequential computation**: Like all RNN-based architectures, the hidden state $h_t$ depends on $h_{t-1}$, preventing parallelization across the time dimension. Training on very long sequences is slow.

**Still limited on very long sequences**: While LSTMs dramatically improve over RNNs, they still struggle with sequences spanning thousands of steps. In those regimes, Transformer architectures with attention mechanisms tend to dominate.

**Forget gate failure mode**: If the forget gate consistently closes ($f_t \approx 0$), the network loses its memory and degrades toward a vanilla RNN. This is a known initialization-sensitive failure, partially mitigated by initializing $b_f$ to a large positive value (e.g., 1.0) so the gate starts open.

**Interpretability**: The four interacting gates, two state vectors, and nonlinear interactions make LSTM internals difficult to interpret. Attributing predictions to specific input time steps requires post-hoc analysis tools (e.g., attention visualization, gradient-based attribution).

**Superseded for many tasks**: In NLP and long-sequence modeling, Transformers have largely replaced LSTMs. LSTMs remain competitive on moderate-length time series where the sequential inductive bias is valuable and training data is limited.

---

## 8. Practical Recommendations

### Forget Gate Initialization

Initialize the forget gate bias $b_f$ to 1 or 2 rather than 0. This ensures the forget gate starts open ($f_t \approx \sigma(1) \approx 0.73$), allowing gradient flow from the beginning of training:

```python
for name, param in model.named_parameters():
    if 'bias_hh' in name or 'bias_ih' in name:
        # Forget gate bias is in positions [H:2H]
        nn.init.constant_(param[hidden_size:2*hidden_size], 1.0)
```

### Gradient Clipping

Even with LSTMs, gradient clipping remains essential:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Regularization

Variational dropout (same dropout mask applied consistently across time steps) is more principled than standard dropout for RNNs, but standard inter-layer dropout in `nn.LSTM` is a practical default.

---

## 9. Summary

1. **LSTMs introduce a cell state** $c_t$ updated via additive operations, providing a near-constant gradient highway that allows information and gradients to flow across hundreds of time steps without exponential decay.

2. **Three gates** — forget, input, and output — are learned functions of the current input and previous hidden state. They implement a differentiable memory management system: what to discard, what to write, and what to expose.

3. **Gradient flow through the cell state** depends only on the forget gate values, not on activation function derivatives. Vanishing is a learnable outcome rather than a structural inevitability.

4. **The cost** of this expressiveness is 4× the parameters of a vanilla RNN, sequential computation, and greater interpretability challenges.

5. **LSTMs are the default choice** for financial time series, speech, and moderate-length sequential tasks. They have been partially displaced by Transformers in NLP and very long sequence modeling, but remain standard in applied time series forecasting.

---

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (1999). Learning to forget: Continual prediction with LSTM. *ICANN*.
- Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. *arXiv*.
- Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A search space odyssey. *IEEE Transactions on Neural Networks and Learning Systems*.

---

**Next Lecture**: Gated Recurrent Unit (GRU)
