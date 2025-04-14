import jax.numpy as jnp

class BallisticStepper:
    g = 9.81
    dt = 0.01
    def __init__(self, cd=0.0):
        self.cd = cd

    def eoms(self, state):
        cd = self.cd
        g = BallisticStepper.g

        x, y, x_dot, y_dot = state
        v_mag = jnp.sqrt(x_dot**2 + y_dot**2)
        x_ddot = -cd * v_mag * x_dot
        y_ddot = -g - cd * v_mag * y_dot

        return jnp.array([x_dot, y_dot, x_ddot, y_ddot])

    def integrate(self, state):
        return state + BallisticStepper.dt*self.eoms(state)

    def __call__(self, state):
        return self.integrate(state)