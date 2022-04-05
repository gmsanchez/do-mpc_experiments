import casadi
import numpy as np
import do_mpc

def template_mhe(model, t_step=1.0, n_horizon=10, P=None, Q=None, R=None):

    # We got no parameters to estimate, so we don't use the second argument
    mhe = do_mpc.estimator.MHE(model)

    setup_mhe = {
        'n_horizon': n_horizon,
        't_step': t_step,
        'store_full_solution': True,
        'nl_cons_check_colloc_points': True,
        'nlpsol_opts': {
        'ipopt.print_level': 0, 'print_time': 0, 'ipopt.suppress_all_output': 'yes'
            # 'ipopt.linear_solver': 'MA27'
        },
    }
    mhe.set_param(**setup_mhe)

    P_x = casadi.diag(model.p['P_x'])
    P_w = casadi.diag(model.p['P_w'])
    P_v = casadi.diag(model.p['P_v'])
    # P_p = model.p['P_p']

    # Set the default MHE objective by passing the weighting matrices:
    mhe.set_default_objective(P_v=P_v, P_x=P_x, P_w=P_w)

    # P_y is listed in the time-varying parameters and must be set.
    # This is more of a proof of concept (P_y is not actually changing over time).
    # We therefore do the following:
    tvp_template = mhe.get_tvp_template()
    # tvp_template['_tvp', :, 'P_v'] = np.diag(np.array([1]))

    # Typically, the values would be reset at each call of tvp_fun.
    # Here we just return the fixed values:
    def tvp_fun(t_now):
        return tvp_template

    mhe.set_tvp_fun(tvp_fun)

    # Only the non estimated parameters must be passed:
    p_template_mhe = mhe.get_p_template()

    def p_fun_mhe(t_now):
        if P is None:
            p_template_mhe['P_x'] = np.eye(model.n_x)
        else:
            p_template_mhe['P_x'] = casadi.diag(casadi.inv(P))
        if Q is None:
            p_template_mhe['P_w'] = np.eye(model.n_x)
        else:
            p_template_mhe['P_w'] = casadi.diag(casadi.inv(Q))
        if R is None:
            p_template_mhe['P_v'] = np.eye(model.n_v)
        else:
            p_template_mhe['P_v'] = casadi.diag(casadi.inv(R))

        return p_template_mhe

    mhe.set_p_fun(p_fun_mhe)

    # Measurement function:
    y_template = mhe.get_y_template()

    def y_fun(t_now):
        n_steps = min(mhe.data._y.shape[0], mhe.n_horizon)
        for k in range(-n_steps, 0):
            y_template['y_meas', k] = mhe.data._y[k]

        return y_template

    mhe.set_y_fun(y_fun)

    mhe.bounds['lower','_x', 'cA'] = 0
    mhe.bounds['lower','_x', 'cB'] = 0
    mhe.bounds['lower','_x', 'cC'] = 0

    mhe.bounds['upper','_x', 'cA'] = np.inf
    mhe.bounds['upper','_x', 'cB'] = np.inf
    mhe.bounds['upper','_x', 'cC'] = np.inf

    mhe.setup()

    return mhe

