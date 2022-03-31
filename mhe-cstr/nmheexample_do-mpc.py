import matplotlib.pyplot as plt
import numpy as np
from casadi import *
import do_mpc

from template_model import template_model
from template_mhe import template_mhe
from template_simulator import template_simulator

t_step = 0.25
Nsim = 80
tplot = np.arange(Nsim+1)*t_step

model = template_model()
simulator = template_simulator(model, t_step=t_step)
# mhe = template_mhe(model, t_step=t_step)

# estimator = do_mpc.estimator.StateFeedback(model)

Nx = model.n_x
Nu = model.n_u
Ny = model.n_y
Nv = Ny
Nw = Nx

x_0 = np.array([1.0, 0.0, 4.0])
x0 = np.array([0.5, 0.05, 0.0])

x_0 = x0

# Simulate
simulator.x0 = x0
u0 = np.array([[0]])

sigma_v = 0.25  # Standard deviation of the measurements
sigma_w = 0.001  # Standard deviation for the process noise
sigma_p = 0.5  # Standard deviation for prior

Q = np.diag((sigma_w*np.ones((Nw,)))**2)
R = np.diag((sigma_v*np.ones((Nv,)))**2)
# R = np.diag([0.5])
# w = sigma_w*np.random.randn(Nsim,Nw)
v = sigma_v*np.random.randn(Nsim,Nv)

usim = np.zeros((Nsim,Nu)) # This is just a dummy input.
xsim = np.zeros((Nsim+1,Nx))
xsim[0,:] = x0
xhat = np.zeros((Nsim+1,Nx))
xhat[0,:] = x0
yclean = np.zeros((Nsim, Ny))
ysim = np.zeros((Nsim, Ny))

N_horizon = 10

# mhe = template_mhe(model, t_step=t_step, R=R)
# N_horizon = mhe.n_horizon
# mhe.x0 = x0
# mhe.set_initial_guess()

# Simulate everything to check if the model works
for t in range(Nsim):
    yclean[t,:] = simulator.make_step(u0)
    ysim[t, :] = yclean[t,:] + v[t,:]
    xsim[t+1,:] = np.squeeze(simulator.x0.cat)
    # xhat[t+1,:] = np.squeeze(mhe.make_step(ysim[t,:]))

    if t <= N_horizon:
        mhe = template_mhe(model, t_step=t_step, n_horizon=t+1, Q=Q, R=R)
        mhe.x0 = x0
        # for i in range(t):
        #     mhe.data.update(_y=yclean[i])
        mhe.set_initial_guess()
        current_measurement = yclean[0:t]
        mhe.data._y = np.reshape(current_measurement, (t, 1))
        # print(mhe.data)
        xhat[t+1,:] = np.squeeze(mhe.make_step(yclean[t,:]))
        print(t, "data:\n", mhe.data._y)
        # xhat[t,:] = np.squeeze(mhe.opt_x_num['_x',0,0])
    else:
        # pass
        xhat[t+1,:] = np.squeeze(mhe.make_step(yclean[t,:]))


    # current_measurement = yclean[t:t+N_horizon-1]
    # mhe.data._y = np.reshape(current_measurement, (N_horizon-1,1))
    # xhat[t+1,:] = np.squeeze(mhe.make_step(yclean[t,:]))
    # xhat[t+1,:] = np.squeeze(mhe.make_step(current_measurement))

    # mhe.data._y = current_measurement
    # mhe.solve()
    # x_next = mhe.opt_x_num['_x', -1, -1]*mhe._x_scaling
    # xhat[t+1,:] = np.squeeze(x_next)

# Plot simulations

plt.figure()

# Plot states
plt.subplot(211)

# Plot states.
colors = ["red","blue","green"]
species = ["A", "B", "C"]


for i in range(Nx):
    plt.plot(tplot, xsim[:, i], label=species[i], color=colors[i])
    plt.plot(tplot, xhat[:, i], marker='o', linestyle='dotted', color=colors[i])

plt.legend()
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.grid()

plt.subplot(212)
plt.plot(yclean, label='$P$', color='black', )
plt.plot(ysim, label='$\hat{P}$', color='black')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.grid()
