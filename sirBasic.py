# A graph of a basic S-I-R Model

# Import libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Basic model variables

N = 1000            # Total population
S0 = 999            # Initial members of Susceptible, 
I0 = 1              # ... infected
R0 = N - S0 - I0    # ... recovered
y0 = S0, I0, R0     # Initial conditions vector

beta = 1            # Beta: product of (contact rate) * (transmission probability)
gamma = 0.1         # Gamma: mean recovery rate. 1/gamma is the average infectious period
t = np.linspace(0, 50, 50) # A grid of time points (in days)

# SIR differential equations.
def diffeq(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * (S / N) * (I / N)
    dIdt = beta * (S / N) * (I / N) - gamma * (I / N)
    dRdt = gamma * (I / N)
    return dSdt, dIdt, dRdt

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

## Plot the data
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
