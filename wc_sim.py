#%%
# INIT
from scipy.integrate import solve_ivp
# ivp := "initial value problem"
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
env_loc="rms_envelopes.pkl"
with open(env_loc,"rb") as file:
  envs=pickle.load(file)

#%%
# Lotka-volterra example from 
# https://ulissigroup.cheme.cmu.edu/math-methods-chemical-engineering/notes/ordinary_differential_equations/18-nonlinear-coupled-ODEs.html
alpha = 1 # 1/day
beta = 0.2 # 1/wolves/day
delta = 0.5 # 1/rabbits/day
gamma = 0.2 # 1/day

def diffeq(t,pop):
  x,y = pop
  return [alpha*x-beta*x*y,
          delta*x*y-gamma*y]

sol = solve_ivp(diffeq, 
                t_span = [0,40],
                y0=[1,5],
                t_eval=np.linspace(0,40,100))

plt.plot(sol.t, sol.y.T,'-')
plt.legend(['Rabbits','Wolves'])
plt.xlabel('Time [days]')
plt.ylabel('Population [#]')
#%%
# Wilson Cowan "classic"
# goal: recreate  Fig S2 in Doelling & Assaneo 2023
 
a=10
b=10
c=10
d=-2 # I think this is the same as g...? typo? or I'm stupid?
#Re: looking at their 2019 paper it does seem they changed d to g for no reason...
# NOTE: gonna use the music paper values since less ambiguous
rho_E=2.3
rho_I=-3.2
tau=66e-3 # in seconds (66 ms)
k=1.5

#
fs=100 #hz
test_env=envs['b01s01p01']
def sigmoid(z):
  return 1/(1+np.exp(-z))
def A(t):
  '''define stimulus envelope here somehow'''
  n=int(round(t*fs))-1
  return test_env[n]
  

def wc_deq(t,neural_activity):
  '''
  Wilson Cowan differential equation
  '''
  E,I=neural_activity
  dEdt=(1/tau)*(-E+sigmoid(rho_E+c*E-a*I+k*A(t)))
  dIdt=(1/tau)*(-I+sigmoid(rho_I+b*E-d*I))
  return [dEdt,
          dIdt]


#%%
# EVAL
t_end=len(test_env)/fs
sol = solve_ivp(wc_deq, 
                t_span = [0,t_end],
                y0=[0.25,0.75],
                t_eval=np.linspace(0,t_end,1000))


#%%
# plot output
E_sol=sol.y[0,:]
I_sol=sol.y[1,:]
out_sol=(E_sol-I_sol).T
t_input=np.arange(len(test_env))/fs
plt.plot(t_input,test_env)
plt.plot(sol.t, out_sol) 
plt.legend(['Speech Envelope','Wilson Cowan Output'])
plt.xlabel('Time (s)')  
plt.title('Wilson Cowan Simulation')
# plt.ylabel('Neural Activity')
#%%
# plot solution
plt.plot(sol.t, sol.y.T,'-') 
plt.legend(['Excitatory', 'Inhibitory'])
plt.xlabel('Time (s)')
plt.ylabel('Neural Activity')