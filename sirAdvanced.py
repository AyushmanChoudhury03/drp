# A graph of a more advanced S-I-R Model

# Import libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Population variables
N = 1000            # Total population
S0 = 990            # Initial members of Susceptible, 
I0 = 10             # ... infected
R0 = 0              # ... recovered
y0 = S0, I0, R0     # Initial conditions vector

# Model variables
t = np.linspace(0, 365, 365)    # A grid of time points (in days)
beta = 50                       # Beta: product of (contact rate) * (transmission probability)
gamma = 5                       # Gamma: mean recovery rate. 1/gamma is the average infectious period
mu = 10                         # Mu: the natural mortality rate (historically, also the birth rate)
rho = 0.3                       # Rho: individual probability of dying from disease
m = rho/(1-rho) * (gamma + mu)  # m: probability of dying from diease before recovering or naturally dying
omega = 0.5                     # Rate at which immunity is lost. 1/omega is the average immunity period

# SIR differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * (S / N) * (I / N) + mu - mu * (S / N) + omega * (R / N)
    dIdt = beta * (S / N) * (I / N) - gamma * (I / N) - mu * (I / N) - m * (I / N)
    dRdt = gamma * (I / N) - mu * (R / N) - omega * (R / N)
    return dSdt, dIdt, dRdt

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T
total = S + I + R

## Plot the data
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.plot(t, total, 'k', label='Total')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
