import jax

# Expose 8 CPU devices for the data-parallel tests. Must run before JAX initializes
# its backend; single-device tests still run on device 0.
jax.config.update("jax_num_cpu_devices", 8)
