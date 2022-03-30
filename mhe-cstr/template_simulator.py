import do_mpc

def template_simulator(model, t_step=1.0):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step=t_step)

    p_template = simulator.get_p_template()
    def p_fun(t_now):
        return p_template
    simulator.set_p_fun(p_fun)

    # The timevarying paramters have no effect on the simulator (they are only part of the cost function).
    # We simply use the default values:
    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator
