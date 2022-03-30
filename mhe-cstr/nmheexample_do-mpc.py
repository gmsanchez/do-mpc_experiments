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
# estimator = do_mpc.estimator.StateFeedback(model)

# mhe = template_mhe(model, t_step=t_step)

Nx = model.n_x
Nu = model.n_u
Ny = model.n_y

x_0 = np.array([1.0, 0.0, 4.0])
x0 = np.array([0.5, 0.05, 0.0])

# Simulate
simulator.x0 = x0
u0 = np.array([[0]])

usim = np.zeros((Nsim,Nu)) # This is just a dummy input.
xsim = np.zeros((Nsim+1,Nx))
xsim[0,:] = x0
yclean = np.zeros((Nsim, Ny))
ysim = np.zeros((Nsim, Ny))


# Simulate everything to check if the model works
for t in range(Nsim):
    yclean[t,:] = simulator.make_step(u0)
    xsim[t+1,:] = np.squeeze(simulator.x0.cat)


# Plot simulations

plt.figure()

# Plot states
plt.subplot(211)

# Plot states.
colors = ["red","blue","green"]
species = ["A", "B", "C"]

for i in range(Nx):
    plt.plot(tplot, xsim[:, i], label=species[i], color=colors[i])

plt.legend()
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.grid()

plt.subplot(212)
plt.plot(yclean, label='P', color='black')
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.grid()
