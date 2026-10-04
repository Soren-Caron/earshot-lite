"""Tiny Keyword Transformer (after Berg et al., "Keyword Transformer", 2021).

Each log-mel frame is a token: linear-embed the 40 mel bins, prepend a learned
CLS token, add learned positional embeddings, run pre-LN transformer blocks,
classify from the CLS token. Shrunk from KWT-1 (12 layers, 607k params) to
3 layers / ~107k params to sit in the same size class as the other models.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Block(nnx.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(heads, dim, dropout_rate=dropout, deterministic=False, decode=False, rngs=rngs)
        self.ln2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        return x + self.dropout(self.fc2(jax.nn.gelu(self.fc1(self.ln2(x)))))


class KWT(nnx.Module):
    def __init__(
        self,
        num_classes: int = 12,
        dim: int = 64,
        depth: int = 3,
        heads: int = 4,
        mlp_dim: int = 128,
        n_frames: int = 49,
        n_mels: int = 40,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed = nnx.Linear(n_mels, dim, rngs=rngs)
        self.cls = nnx.Param(jnp.zeros((1, 1, dim)))
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, n_frames + 1, dim)))
        self.blocks = nnx.List([Block(dim, heads, mlp_dim, dropout, rngs=rngs) for _ in range(depth)])
        self.ln = nnx.LayerNorm(dim, rngs=rngs)
        self.head = nnx.Linear(dim, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.embed(x[..., 0])  # (B, T, F, 1) -> (B, T, dim)
        cls = jnp.broadcast_to(self.cls[...], (x.shape[0], 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1) + self.pos[...]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x)[:, 0])
