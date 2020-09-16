import nose.tools as nt       # Using nosetests
import numpy as np
from pendulum import Pendulum


def test_special_method_call_in_Pendulum_class():
    """Tests that theta and omega is computed correctly."""
    # Test values
    theta    = np.pi/4 # Angular position of the pendulum
    omega    = 0.1     # Angular velocity of the pendulum
    analytic = [0.1, -3.1530534197454685]
    eps      = 10**(-7)

    pendel   = Pendulum(L=2.2)
    computed = pendel(0,[theta,omega])

    assert(abs(computed[0] - analytic[0]) < eps)
    assert(abs(computed[1] - analytic[1]) < eps)


def test_special_method_call_in_Pendulum_class_keeps_a_peldelum_at_rest():
    """Tests that the pendulum is kept at rest."""
    # Test values
    theta0   = 0
    omega0   = 0
    analytic = [0, 0]
    eps      = 10**(-7)

    pendel   = Pendulum()
    computed = pendel(0,[theta0,omega0])

    assert(abs(computed[0] - analytic[0]) < eps)
    assert(abs(computed[1] - analytic[1]) < eps)


@nt.raises(AttributeError)
def test_error_if_solve_method_has_not_been_called():
    """
    Test that the solve method has been called. Error raised if attributes dont exist.
    """
    pendel = Pendulum()
    theta  = pendel.theta
    omega  = pendel.omega
    time   = pendel.t


def test_only_the_latest_solution_is_stored():
    """Tests that latest solution overwrites previous ones."""
    y0_1 = [0, 0]
    T_1  = 5
    dt_1 = 0.1

    y0_2 = [2, 3]
    T_2  = 15
    dt_2 = 0.01

    y0_3 = [1, 4]
    T_3  = 10
    dt_3 = 0.05

    pendel = Pendulum()
    pendel.solve(y0_1, T_1, dt_1)
    len_1 = len(pendel.t) #store previous length
    pendel.solve(y0_2, T_2, dt_2)
    len_2 = len(pendel.t) #store previous length
    pendel.solve(y0_3, T_3, dt_3)
    #Check length of t
    assert(len(pendel.t) != len_1)
    assert(len(pendel.t) != len_2)

    pendel2 = Pendulum()
    pendel2.solve(y0_3, T_3, dt_3)
    # Solve pendel2 for case #3 only
    # Check so that pendel is the latest solution
    for i in range(len(pendel.x)):
        assert(pendel.x[i] == pendel2.x[i])
        assert(pendel.y[i] == pendel2.y[i])


def test_solve_method_in_Pendulum_class_theta_omega_zero_arrays():
    """
    Test solve method keeps pendulum at rest for initial y0=[0,0] while t=i*dt.
    """
    y0 = [0, 0]
    T  = 5
    dt = 0.1

    pendel = Pendulum()
    pendel.solve(y0, T, dt)

    for i in range(len(pendel.t)):
        assert(pendel.t[i]     == i*pendel.dt)
        assert(pendel.theta[i] == 0)
        assert(pendel.omega[i] == 0)


def test_x_and_y_positions_are_correct():
    """
    Tests that x and y position is computed correctly by testing x^2 + y^2 = L^2.
    """
    y0  = [2, 3]
    T   = 15
    dt  = 0.01
    eps = 10**(-7)

    pendel = Pendulum(L=2)
    pendel.solve(y0, T, dt)
    sol = pendel.solution

    array_of_L = np.zeros(len(sol.y[0])) + (pendel.L**2)
    computed_radius_squared = pendel.x**2 + pendel.y**2
    for i in range(len(sol.y[0])):
        assert(abs(computed_radius_squared[i] - array_of_L[i]) < eps)

if __name__ == "__main__":
    import nose
    nose.run()
