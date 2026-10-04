"""Stacked unidirectional GRU over log-mel frames.

Each of the 49 frames (40 mel bins) is one time step; the final hidden state of
the top layer feeds a linear classifier. Unidirectional so it could run
streaming, one frame at a time, the way an always-on detector would.
"""

from flax import nnx


class GRUKWS(nnx.Module):
    def __init__(self, num_classes: int = 12, hidden: int = 64, num_layers: int = 2, n_mels: int = 40, dropout: float = 0.1, *, rngs: nnx.Rngs):
        self.rnns = nnx.List(
            [nnx.RNN(nnx.GRUCell(n_mels if i == 0 else hidden, hidden, rngs=rngs)) for i in range(num_layers)]
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.head = nnx.Linear(hidden, num_classes, rngs=rngs)

    def __call__(self, x):
        x = x[..., 0]  # (B, T, F, 1) -> (B, T, F)
        for rnn in self.rnns:
            x = rnn(x)
        return self.head(self.dropout(x[:, -1]))
