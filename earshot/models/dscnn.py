"""Depthwise-separable CNN for keyword spotting (Zhang et al., "Hello Edge", 2017).

    conv 10x4 /2 -> BN -> ReLU
    N x [ depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU ]
    global avg pool -> dropout -> linear

Convs have no bias because BN follows; BN gets folded into the conv weights
before int8 quantization. Input layout is NHWC = (batch, time, mel, 1).
"""

import jax
from flax import nnx


class ConvBN(nnx.Module):
    def __init__(self, cin, cout, kernel, strides=1, groups=1, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            cin, cout, kernel, strides, padding="SAME", feature_group_count=groups, use_bias=False, rngs=rngs
        )
        self.bn = nnx.BatchNorm(cout, momentum=0.9, rngs=rngs)

    def __call__(self, x):
        return jax.nn.relu(self.bn(self.conv(x)))


class DSCNN(nnx.Module):
    def __init__(self, num_classes: int = 12, channels: int = 64, num_blocks: int = 4, dropout: float = 0.2, *, rngs: nnx.Rngs):
        c = channels
        self.stem = ConvBN(1, c, (10, 4), strides=(2, 2), rngs=rngs)
        self.blocks = nnx.List(
            [
                nnx.Sequential(ConvBN(c, c, (3, 3), groups=c, rngs=rngs), ConvBN(c, c, (1, 1), rngs=rngs))
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.head = nnx.Linear(c, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(axis=(1, 2))
        return self.head(self.dropout(x))
