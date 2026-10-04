"""Count multiply-accumulates (MACs) of a JAX function by walking its jaxpr.

Counts dot_general and conv_general_dilated (the ops that dominate KWS models),
multiplies scan bodies by their trip count, and recurses into nested jaxprs
(pjit, custom_jvp/vjp, remat, cond). XLA's own cost_analysis counts a scan body
only once, which undercounts RNNs by ~T x. Elementwise ops are ignored.
"""

import math

import jax
from flax import nnx


def _sub_jaxprs(params):
    for v in params.values():
        for item in v if isinstance(v, (tuple, list)) else (v,):
            if isinstance(item, jax.extend.core.ClosedJaxpr):
                yield item.jaxpr
            elif isinstance(item, jax.extend.core.Jaxpr):
                yield item


def _count(jaxpr) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "dot_general":
            (lhs_contract, _), _ = eqn.params["dimension_numbers"]
            lhs = eqn.invars[0].aval.shape
            total += math.prod(eqn.outvars[0].aval.shape) * math.prod(lhs[d] for d in lhs_contract)
        elif name == "conv_general_dilated":
            rhs = eqn.invars[1].aval.shape
            out_ch_dim = eqn.params["dimension_numbers"].rhs_spec[0]
            total += math.prod(eqn.outvars[0].aval.shape) * math.prod(rhs) // rhs[out_ch_dim]
        elif name == "scan":
            total += eqn.params["length"] * _count(eqn.params["jaxpr"].jaxpr)
        elif name == "cond":
            total += max(_count(b.jaxpr) for b in eqn.params["branches"])
        elif name == "while":
            raise NotImplementedError("while loops have no static trip count")
        else:
            total += sum(_count(j) for j in _sub_jaxprs(eqn.params))
    return total


def count_macs(fn, *args) -> int:
    return _count(jax.make_jaxpr(fn)(*args).jaxpr)


def model_macs(model: nnx.Module, input_shape=(1, 49, 40, 1)) -> int:
    """MACs for one forward pass of an nnx model in eval mode (per batch of input_shape)."""
    model.eval()
    graphdef, state = nnx.split(model)
    # state is traced (not closed over) so modules that advance RNG counters, e.g. nnx.RNN, can be traced
    return count_macs(lambda s, x: nnx.merge(graphdef, s)(x), state, jax.numpy.zeros(input_shape))
