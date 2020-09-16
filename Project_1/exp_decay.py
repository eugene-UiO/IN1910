import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class ExponentialDecay():
    """Exponential decay class."""
    def __init__(self, a):
        """ExponentialDecay class takes a decay constant a as argument."""
        self.a = a


    def __call__(self, t, u):
        """
        Special method __call__ that finds the derivative of u when u(t) is
        known.
        Parameters: t -- Time where the derivative should be calculated at.
                    u -- Value of the decay function at the given time t.
        Returns: The derivative of the decay function u at a time t.
        """
        return -self.a*u


    def solve(self, u0, T, dt):
        """
        Solves the ODE using scipy.integrate solve_ivp function.
        Parameters: u0   -- An initial value for the function u(0) at time t=0.
                    T    -- End time for the experiment.
                    dt   -- Timestep we want to use in our calculation.
        Returns: sol.t -- Array with time values.
                 sol.y -- Array with values of u at given time t in sol.t
        """
        sol = solve_ivp(self, [0, T], [u0], t_eval=np.arange(0, T, dt))
        return sol.t, sol.y[0]


if __name__ == "__main__":
    #plotting a test run
    a  = 0.4    # Decay constants
    u0 = 3.2    # Function value u(t) for some known time t
    T  = 5.     # Timespan
    dt = 0.1    # Timestep

    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)
    plt.style.use("classic")
    plt.plot(t, u)
    plt.grid()
    plt.show()
