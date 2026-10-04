"""Data-parallel helpers (SPMD via jax.sharding).

The model and optimizer state are replicated on every device; each batch is
split along a 1-D mesh axis "data". The unchanged jitted train step is then
partitioned by XLA, which inserts the gradient all-reduce itself. BatchNorm
statistics are reduced over the global batch, so N devices compute the same
math as one device with the full batch (synchronized BatchNorm for free).

The same code runs on 8 simulated CPU devices (`simulate_cpu_devices(8)`,
before JAX initializes) and on a real 8-core TPU.
"""

import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

DATA = "data"


def simulate_cpu_devices(n: int) -> None:
    """Expose n CPU devices. Must run before anything initializes the JAX backend."""
    jax.config.update("jax_num_cpu_devices", n)


def data_mesh(n_devices: int | None = None) -> Mesh:
    devices = jax.devices()
    n = n_devices or len(devices)
    if n > len(devices):
        raise ValueError(f"asked for {n} devices, only {len(devices)} available: {devices}")
    return Mesh(np.array(devices[:n]), (DATA,))


def replicate(mesh: Mesh, *nodes) -> None:
    """Place the state of nnx objects (model, optimizer, ...) fully replicated on the mesh, in place."""
    state = nnx.state(nodes)
    nnx.update(nodes, jax.device_put(state, NamedSharding(mesh, P())))


def shard_batch(mesh: Mesh, *arrays):
    """Split host arrays along their leading (batch) axis across the mesh."""
    sharding = NamedSharding(mesh, P(DATA))
    out = tuple(jax.device_put(a, sharding) for a in arrays)
    return out if len(out) > 1 else out[0]
