import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


class Pendulum():
    """
    Class for modeling a pendulum with a given mass M, hanging in a massless
    rod with length L, affected by a gravitational force pointing directly down
    at all time. In other words, a normal pendulum.
    """


    def __init__(self, L=1, M=1, g=9.81):
        """
        The class should be called with the following arguments:
        Paramenters: M -- Pendulum mass [kg]
                     L -- Length of massless rod [m]
                     g -- Gravitational acceleration [m/s^2]
        """
        self.L = L
        self.M = M
        self.g = g


    def __call__(self, t, y):
        """
        __call__ special method for caluculating the derivative of the angle
        theta [angle(unitless)] and the angular velocity omega [1/s] for a
        single pendulum.
        Paramenters: t -- Time [s].
                     y -- Tuple consisting of initial values for theta [] and
                          omega [1/s]. Should be given as a tuple
                          (theta, omega).
        Returns: Derivative of angle and angular momentum as a tuple in that
                 order.
        """
        d_theta__dt = y[1]                          # Angular velocity omega
        d_omega__dt = -(self.g/self.L)*np.sin(y[0]) # Angular acceleration
        return d_theta__dt, d_omega__dt


    def solve(self, y0, T, dt, angles="rad"):
        """
        Solves the ODE using scipy.integrate solve_ivp function.
        Parameters: y0   -- An initial values for the function y we want to find
                            at a time t=0. Should be given as a list
                            [theta0, omega0].
                    T    -- End time for the experiment.
                    dt   -- Timestep we want to use in our calculation.
                    (NB! Both T and dt are given as scalar values. That is,
                    solve method should be called as
                    object.solve([theta0, omega0], T, dt), where theta0, omega0,
                    T and dt are scalar values.)
        Kwargs: angles -- Specifying if the given values of theta and omega
                          are given in degrees or radians. Should be specified
                          as "deg" or "rad". If angles="deg", it will be changed
                          to radians. The result will allways be given in
                          radians.
        Returns: Nothing. Sets the solution as a private variable _solution.
        """
        self._dt = dt
        self._T =T


        if angles == "deg":
            y0 = np.array(y0)
            y0 = y0*(np.pi/180) # Changing from degrees to radians.

        sol = solve_ivp(self, [0, T], y0, t_eval=np.arange(0, T, dt),method="Radau")
        self._solution = sol


    @property
    def solution(self):
        """Returns the latest stored solution."""
        try:
            return self._solution
        except AttributeError:
            raise AttributeError("Solutions for the pendulum does not exist. "
                                 "You need to run the solve method to compute "
                                 "how the pendulum moves. Run solve method "
                                 "first.")


    @property
    def t(self):
        """
        Returns the time calculated in solve method. Raises error if it doesn't
        exit.
        """
        try:
            return self.solution.t
        except AttributeError:
            raise AttributeError("Pendulum time does not exist. You need to "
                                 "run the solve method to compute a time. Run "
                                 "solve method first.")


    @property
    def theta(self):
        """Returns angle, raises error if it doesn't exist."""
        try:
            return self.solution.y[0]
        except AttributeError:
            raise AttributeError("Theta values for pendulum does not exist "
                                 "yet. You need to run the solve method to "
                                 "compute the theta values. Run solve method "
                                 "first.")


    @property
    def omega(self):
        """Returns angular velocity, raises error if it doesn't exist."""
        try:
            return self.solution.y[1]
        except AttributeError:
            raise AttributeError("Omega values for pendulum does not exist "
                                 "yet. You need to run the solve method to "
                                 "compute the omega values. Run solve method "
                                 "first.")


    @property
    def dt(self):
        """Returns dt as given in solve."""
        return self._dt


    @property
    def T(self):
        """Returns T as given in solve."""
        return self._T


    @property
    def x(self):
        """Returns x position."""
        x_pos = self.L*np.sin(self.theta)
        return x_pos


    @property
    def y(self):
        """Returns y position."""
        y_pos = -self.L*np.cos(self.theta)
        return y_pos


    @property
    def vx(self):
        """Returns speed in x direction."""
        return np.gradient(self.x,self.t)


    @property
    def vy(self):
        """Returns speed in y direction"""
        return np.gradient(self.y,self.t)


    @property
    def potential(self):
        """Returns calculated potential energy"""
        return self.M*self.g*(self.y + self.L)


    @property
    def kinetic(self):
        """Returns calculated kinetic energy"""
        return 0.5*self.M*(self.vx**2 + self.vy**2)


class DampenedPendulum(Pendulum):


    """Class for modeling a dampened pendulum."""
    def __init__(self, L=1, M=1, g=9.81, B=0):
        """
        The class should be called with the following arguments:
        Paramenters: M -- Pendulum mass [kg]
                     L -- Length of massless rod [m]
                     g -- Gravitational acceleration [m/s^2]
                     B -- Damping factor [kg/s]
        """
        self.L = L
        self.M = M
        self.g = g
        self.B = B


    def __call__(self, t, y):
        """
        __call__ special method for caluculating the derivative of the angle
        theta [angle(unitless)] and the angular velocity omega [1/s].
        Paramenters: t    -- Time [s]
                     y    -- Tuple consisting of initial values for theta []
                             and omega [1/s]. Should be given as a tuple
                             (theta, omega).
        Returns: Derivative of angle and angular momentum.
        """
        d_theta__dt = y[1]                                               # Angular velocity omega
        d_omega__dt = -(self.g/self.L)*np.sin(y[0])-(self.B/self.M)*y[1] # Angular acceleration
        return d_theta__dt, d_omega__dt


if __name__ == "__main__":
    #Plotting regular penulum, movement and energy levels
    u0 = [np.pi/3, 0.423]   # [theta,omega]
    T  = 10                 # Timespan
    dt = 0.01               # Timestep

    pendel = Pendulum(M=5, L=3.2)
    pendel.solve(u0,T,dt)

    time  = pendel.t
    theta = pendel.theta
    plt.style.use("classic")
    plt.grid()
    plt.plot(time, theta)
    plt.title("Regular pendulum")
    plt.axis("equal")
    plt.xlabel("time [s]") ; plt.ylabel("Angle theta [rad]")
    plt.show()

    Ke = pendel.kinetic
    Pe = pendel.potential
    Te = Ke + Pe

    plt.grid()
    plt.plot(time,Ke)
    plt.plot(time,Pe)
    plt.plot(time,Te)
    plt.title("Regular pendulum")
    plt.legend(["Kinetic energy","Potential energy","Total energy"])
    plt.xlabel("time [s]") ; plt.ylabel("Energy [J]")
    plt.show()

    #Plotting dampened pendulum, movement and energy levels
    dpendel = DampenedPendulum(M=5, L=3.2, B=1.5)
    dpendel.solve(u0,T,dt)

    damp_time  = dpendel.t
    damp_theta = dpendel.theta
    plt.grid()
    plt.plot(time, dpendel.theta)
    plt.axis("equal")
    plt.title("Dampened pendulum")
    plt.xlabel("time [s]")
    plt.ylabel("Angle theta [rad]")
    plt.show()

    damp_Ke = dpendel.kinetic
    damp_Pe = dpendel.potential
    damp_Te = damp_Pe + damp_Ke

    plt.plot(damp_time,damp_Ke)
    plt.plot(damp_time,damp_Pe)
    plt.plot(damp_time,damp_Te)
    plt.title("Dampened pendulum")
    plt.legend(["Kinetic energy","Potential energy","Total energy"])
    plt.xlabel("time [s]") ; plt.ylabel("Energy [J]")
    plt.grid()
    plt.show()
