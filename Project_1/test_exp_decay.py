from exp_decay import ExponentialDecay


def test_call_function_ExponentialDecay():
    """
    Test function for the special method __call__ in the ExponentialDecay class.
    """
    a     = 0.4     # Decay constant
    u0    = 3.2     # Function value u(t) for some known time t
    der_u = -1.28   # Analytic value for the derivative of u at the known time t
    eps   = 10**(-7)# Since we are dealing with floating point numbers,
                    # we need a limit when checking that a difference is zero.
    decay_model = ExponentialDecay(a)
    assert(abs(decay_model(0, u0)-der_u) < eps)


if __name__ == "__main__":
    import nose
    nose.run()
