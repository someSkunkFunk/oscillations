#%%
# INIT
# see Lotka-volterra example from 
# https://ulissigroup.cheme.cmu.edu/math-methods-chemical-engineering/notes/ordinary_differential_equations/18-nonlinear-coupled-ODEs.html
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
ks=[1.5, 100, 0.03, 7]
#
fs=100 #hz
# first sentence as test (already downsampled to 100 Hz)
test_env=envs['b01s01p01']
t_end=len(test_env)/fs # end time
t_input=np.arange(len(test_env))/fs # input time vector for plotting
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

#set manually by plotting them and seeing which would give enough space to actually see the lines

ylims=[[(-0.75,1),(0,1.3)],[(-0.75,0.75),(0,1.1)],[(-0.7,1),(0,1.1)],[(-0.8,1),(0,1.3)]]
for i_k, k in enumerate(ks):
  print(f'solving for k={k}...')
  sol=solve_ivp(wc_deq, 
                  t_span = [0,t_end],
                  y0=[0.25,0.75],
                  t_eval=np.linspace(0,t_end,1000))

  E_sol=sol.y[0,:]
  I_sol=sol.y[1,:]
  out_sol=(E_sol-I_sol).T

  # plot output
  fig, axs=plt.subplots(2,1,sharex=True,figsize=(16,12))
  
  axs[0].plot(t_input,test_env)
  axs[0].plot(sol.t, out_sol) 
  axs[0].legend(['Speech Envelope','Wilson Cowan Output (E-I)'])
  # axs[0].set_xlabel('Time (s)') 
  axs[0].set_title(f'Wilson Cowan Simulation k={k}')
  axs[0].set_ylim(ylims[i_k][0])
  # plt.ylabel('Neural Activity')

  # plot solution
  axs[1].plot(sol.t, sol.y.T,'-') 
  axs[1].legend(['Excitatory', 'Inhibitory'])
  axs[1].set_xlabel('Time (s)')
  axs[1].set_ylabel('Neural Activity (au)')
  axs[1].set_title(f'Wilson Cowan simulated response, k={k}')
  axs[1].set_ylim(ylims[i_k][1])