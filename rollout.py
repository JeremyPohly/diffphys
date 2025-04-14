import jax
import jax.numpy as jnp

def rollout(stepper, n_steps: int, include_init: bool = False):
    """
    Roll out the dynamics over a given time array.

    Args:
        stepper: Function to compute the next state, accepts (state).
        time_array: Array of time points.
        include_init: Whether to include the initial state in the trajectory.

    Returns:
        Trajectory of states as a JAX array.
    """
    def scan_fn(carry, _):
        state, done = carry
        next_state = stepper(state)
        new_done = jnp.logical_or(done, next_state[1] < 0.0)
        next_state = jnp.where(done, state, next_state)
        return (next_state, new_done), next_state 

    def rollout_fn(state_init):
        init_done = False
        (final_state, _), trj = jax.lax.scan(scan_fn, (state_init, init_done),
            xs=None, length=n_steps)

        if include_init:
            return jnp.concatenate(
                [jnp.expand_dims(state_init, axis=0), trj], axis=0)

        return trj

    return jax.jit(rollout_fn)
