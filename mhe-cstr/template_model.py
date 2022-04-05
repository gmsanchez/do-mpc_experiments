import do_mpc

def template_model(symvar_type='SX'):

    model_type = 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Parameters of the system
    k1 = 0.5
    k_1 = 0.05
    k2 = 0.2
    k_2 = 0.01
    RT = 32.84


    # States of the system
    cA = model.set_variable(var_type='_x', var_name='cA', shape=(1,1))
    cB = model.set_variable(var_type='_x', var_name='cB', shape=(1,1))
    cC = model.set_variable(var_type='_x', var_name='cC', shape=(1,1))

    # Dummy control imput
    u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    # Arrival cost

    # Parameter for the MHE: Weighting of the arrival cost (parameters):
    P_x = model.set_variable(var_type='_p', var_name='P_x', shape=(3,1))
    # P_p = model.set_variable(var_type='_p', var_name='P_p', shape=(1,1))

    P_w = model.set_variable(var_type='_p', var_name='P_w', shape=(3,1))
    # Time-varying parameter for the MHE: Weighting of the measurements (tvp):
    # Not time varying yet
    P_v = model.set_variable(var_type='_p', var_name='P_v', shape=(1, 1))


    # Measurements of the system
    measurements = RT * (cA + cB + cC)

    model.set_meas('P_meas', measurements, meas_noise=True)

    # Control input measurement
    # model.set_meas('u_meas', u, meas_noise=False)

    # RHS
    rate1 = k1*cA - k_1*cB*cC
    rate2 = k2*cB**2 - k_2*cC

    has_process_noise = True  # if set to True, we get some errors
    model.set_rhs('cA', -rate1, process_noise=has_process_noise)
    model.set_rhs('cB', rate1 - 2*rate2, process_noise=has_process_noise)
    model.set_rhs('cC', rate1 + rate2, process_noise=has_process_noise)

    model.setup()

    return model
