import numpy as np
from numpy.random import randn
import random
from scipy.optimize import minimize
from scipy import interpolate
np.set_printoptions(precision=4, suppress=True)
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from numpy import linalg as LA

def gen_params():
    beta = 0.99  # discount factor
    gamma = 1  # utility function parameter
    alpha = 0.36  # share of capital in production function
    delta = 0.025  # depreciation rate
    mu = 0.15  # unemployment benefits as a share of the wage
    l_bar = 1/0.9  # time endowment; normalizes labor supply to 1 in bad state
    k_ss = ((1/beta-(1-delta))/alpha)**(1/(alpha-1))
    return alpha, beta, gamma, delta, mu, l_bar, k_ss


def gen_grid():
    N = 5000  # number of agents for stochastic simulation
    J = 1000  # number of grid points for stochastic simulation
    k_min = 0
    k_max = 100
    burn_in = 1000
    T = 10000 + burn_in
    ngridk = 80
    x = np.linspace(0, 0.5, ngridk)
    tau = 3
    y = (x/np.max(x))**tau # polynomial grid
    km_min = 10
    km_max = 50
    k = k_min + (k_max-k_min)*y
    ngridkm = 7
    km = np.linspace(km_min, km_max, ngridkm)
    return (N, J, k_min, k_max, T, burn_in, k, km_min, km_max,  km,
            ngridk, ngridkm)

def shocks_parameters():
    nstates_id = 2    # number of states for the idiosyncratic shock
    nstates_ag = 2    # number of states for the aggregate shock
    ur_b = 0.1        # unemployment rate in a bad aggregate state
    er_b = (1-ur_b)   # employment rate in a bad aggregate state
    ur_g = 0.04       # unemployment rate in a good aggregate state
    er_g = (1-ur_g)   # employment rate in a good aggregate state
    epsilon = np.arange(0, nstates_id)
    delta_a = 0.01
    a = np.array((1-delta_a, 1+delta_a))
    prob = np.array(([0.525, 0.35, 0.03125, 0.09375],
                 [0.038889, 0.836111, 0.002083, 0.122917],
                 [0.09375, 0.03125, 0.291667, 0.583333],
                 [0.009115, 0.115885, 0.024306, 0.850694]))
    return nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob

def shocks():
    (N, J, k_min, k_max, T, burn_in, k, km_min, km_max,  km, ngridk, ngridkm) = gen_grid()
    nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()
    ag_shock = np.zeros((T, 1))
    id_shock = np.zeros((T, N))
    np.random.seed(0)
    # number of employment for both good and bad times
    u_g = int(er_g * N)
    u_b = int(er_b * N) 

    # Transition probabilities between aggregate states
    prob_ag = np.zeros((2, 2))
    prob_ag[0, 0] = prob[0, 0]+prob[0, 1]
    prob_ag[1, 0] = 1-prob_ag[0, 0] # bad state to good state
    prob_ag[1, 1] = prob[2, 2]+prob[2, 3]
    prob_ag[0, 1] = 1-prob_ag[1, 1]

    P = prob/np.kron(prob_ag, np.ones((2, 2)))
    # generate aggregate shocks
    mc = qe.MarkovChain(prob_ag)
    ag_shock = mc.simulate(ts_length=T, init=0)  # start from bad state
    # generate idiosyncratic shocks for all agents in the first period
    draw = np.random.uniform(size=N)
    id_shock[0, :] = draw>ur_b #set state to good if probability exceeds ur_b

    # Function used to modified the number of employment during bad and good state
    def adjustUnemploymentNumber(L, S):
        n = len(L)
        if S == 0:
            # bad state
            while sum(L) > u_b:
                r = random.randint(0,n-1)
                if L[r] == 1:
                    L[r] = 0
            while sum(L) < u_b:
                r = random.randint(0,n-1)
                if L[r] == 0:
                    L[r] = 1
        else:
            while sum(L) > u_g:
                r = random.randint(0,n-1)
                if L[r] == 1:
                    L[r] = 0
            while sum(L) < u_g:
                r = random.randint(0,n-1)
                if L[r] == 0:
                    L[r] = 1      
        return L
    # generate idiosyncratic shocks for all agents starting in second period
    draw = np.random.uniform(size=(T-1, N))
    adjustUnemploymentNumber(id_shock[0, :], ag_shock[0])
    for t in range(1, T):
        if t%1000 == 0:
            print(t)
        # Fix idiosyncratic itransition matrix conditional on aggregate state
        transition = P[2*ag_shock[t-1]: 2*ag_shock[t-1]+2, 2*ag_shock[t]: 2*ag_shock[t]+2]
        transition_prob = [transition[int(id_shock[t-1, i]), int(id_shock[t-1, i])] for i in range(N)]
        check = transition_prob>draw[t-1, :] #sign whether to remain in current state
        id_shock[t, :] = id_shock[t-1, :]*check + (1-id_shock[t-1, :])*(1-check)
        adjustUnemploymentNumber(id_shock[t, :], ag_shock[t])
    return id_shock, ag_shock


# function of low of motion
def H(S, k_bar):
    global a0, a1, b0, b1
    # Return k_bar_prime
    if S == 0:
        return np.exp(a0 + a1*np.log(k_bar))
    else:
        return np.exp(b0 + b1*np.log(k_bar))
    
# function of wage
def w(k_bar, L_bar, z, alpha):
    return (1-alpha)*z*(k_bar/L_bar)**alpha

# function of investment return
def r(k_bar, L_bar, z, alpha):
    return alpha*z*(k_bar/L_bar)**(alpha-1)

# value function 
def v(k1, k_bar, S, eps, k, km, V00, V01, V10, V11):
    if S == 0 and eps == 0:
        return float(interpolate.RectBivariateSpline(k, km, V00, kx=3, ky=2)(k1, k_bar))
    elif S == 0 and eps == 1:
        return float(interpolate.RectBivariateSpline(k, km, V01, kx=3, ky=2)(k1, k_bar))
    elif S == 1 and eps == 0:
        return float(interpolate.RectBivariateSpline(k, km, V10, kx=3, ky=2)(k1, k_bar))
    else:
        return float(interpolate.RectBivariateSpline(k, km, V11, kx=3, ky=2)(k1, k_bar))
    
# utility function
def U(c):
    if c <= 0:
        return -1000000
    else:
        return np.log(c)

# objective function
def obj(k_prime):
    global k1, k_bar, L_bar, z, l_bar, alpha, beta, delta, S, eps, k, km, V00, V01, V10, V11
    prob = np.array(([0.525, 0.35, 0.03125, 0.09375],
                 [0.038889, 0.836111, 0.002083, 0.122917],
                 [0.09375, 0.03125, 0.291667, 0.583333],
                 [0.009115, 0.115885, 0.024306, 0.850694]))
    # calculate conditional expectation value
    v00 = v(k_prime, H(S, k_bar), 0, 0, k, km, V00, V01, V10, V11)
    v01 = v(k_prime, H(S, k_bar), 0, 1, k, km, V00, V01, V10, V11)
    v10 = v(k_prime, H(S, k_bar), 1, 0, k, km, V00, V01, V10, V11)
    v11 = v(k_prime, H(S, k_bar), 1, 1, k, km, V00, V01, V10, V11)
    # for simplification
    if S == 0 and eps == 0:
        E = np.dot(prob[0],[v00, v01, v10, v11])
    elif S == 0 and eps == 1:
        E = np.dot(prob[1],[v00, v01, v10, v11])
    elif S == 1 and eps == 0:
        E = np.dot(prob[2],[v00, v01, v10, v11])
    else:
        E = np.dot(prob[3],[v00, v01, v10, v11])
    # Value of consumption    
    c = r(k_bar, L_bar, z, alpha)*k1 + w(k_bar, L_bar, z, alpha)*l_bar*eps + \
      (1-delta)*k1 - k_prime    
    return -(U(c) + beta*E)

(N, J, k_min, k_max, T, burn_in, k, km_min, km_max,  km,
            ngridk, ngridkm) = gen_grid()
(alpha, beta, gamma, delta, mu, l_bar, k_ss) = gen_params()
I,J = len(k),len(km)
# For value function grid
V00 = np.zeros((I,J))
V01 = np.zeros((I,J))
V10 = np.zeros((I,J))
V11 = np.zeros((I,J))
V00_new = np.zeros((I,J))
V01_new = np.zeros((I,J))
V10_new = np.zeros((I,J))
V11_new = np.zeros((I,J))

k00 = np.zeros((I,J))
k01 = np.zeros((I,J))
k10 = np.zeros((I,J))
k11 = np.zeros((I,J))
k00_new = np.zeros((I,J))
k01_new = np.zeros((I,J))
k10_new = np.zeros((I,J))
k11_new = np.zeros((I,J))


def sovleIndividual(tol = 0.001):
    global k1, k_bar, L_bar, z, l_bar, alpha, beta, delta, S, eps, k, km, \
    V00, V01, V10, V11, k00, k01, k10, k11, a0, a1, b0, b1
    # return capital in the next step
    # Control interation times
    count = 0
    # Solve the DDP by value interation
    while True:
        if count == 2000:
            break
        for i in range(I):
            for j in range(J):
                k1 = k[i]
                k_bar = km[j]
                # Four different State
                for S in [0,1]:
                    if S==0:
                        z = a[0]
                        L_bar = 0.9 * l_bar
                    else:
                        z = a[1]
                        L_bar = 0.96 * l_bar
                    for eps in [0,1]:
                        if S == 0 and eps == 0:
                            res = minimize(obj, x0 = 1, bounds = [(0,None)])
                            k00_new[i,j] = res.x
                            V00_new[i,j] = -res.fun
                        elif S == 0 and eps == 1:
                            res = minimize(obj, x0 = 1, bounds = [(0,None)])
                            k01_new[i,j] = res.x
                            V01_new[i,j] = -res.fun
                        elif S == 1 and eps == 0:
                            res = minimize(obj, x0 = 1, bounds = [(0,None)])
                            k10_new[i,j] = res.x
                            V10_new[i,j] = -res.fun
                        else:
                            res = minimize(obj, x0 = 1, bounds = [(0,None)])
                            k11_new[i,j] = res.x
                            V11_new[i,j] = -res.fun

        vdiff00 = LA.norm(V00 - V00_new)
        vdiff01 = LA.norm(V01 - V01_new)
        vdiff10 = LA.norm(V10 - V10_new)
        vdiff11 = LA.norm(V11 - V11_new)

        print("Value function matrix norm difference at each step: ", vdiff00, vdiff01, vdiff10, vdiff11)

        kdiff00 = LA.norm(k00 - k00_new)
        kdiff01 = LA.norm(k01 - k01_new)
        kdiff10 = LA.norm(k10 - k10_new)
        kdiff11 = LA.norm(k11 - k11_new)
        
        print("k value matrix norm difference:", kdiff00, kdiff01, kdiff10, kdiff11)

        if kdiff00 < tol and kdiff01 < tol and kdiff10 < tol and kdiff11 < tol and count > 3:
            break
        else:
            V00 = np.copy(V00_new)
            V01 = np.copy(V01_new)
            V10 = np.copy(V10_new)
            V11 = np.copy(V11_new)
            
            k00 = np.copy(k00_new)
            k01 = np.copy(k01_new)
            k10 = np.copy(k10_new)
            k11 = np.copy(k11_new)
            count += 1

a0 = 0.095
a1 = 0.962
b0 = 0.085
b1 = 0.965
# Global variable
k_bar = km[0]
k1 = k[0]
delta_a = 0.01
a = np.array((1-delta_a, 1+delta_a))
L_bar = 0.96 * l_bar
z = a[0]
S = 0
eps = 0
sovleIndividual()
