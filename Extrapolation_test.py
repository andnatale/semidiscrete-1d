"""Test for metric extrapolation problem"""
import numpy as np
import matplotlib.pyplot as plt
from src.ExtrapolationOneDim import Extrapolation1D

N = 10000 
L = 1.
X = np.linspace(1/2/N,L-1/2/N,N)


# TEST EXTRAPOLATION

#c=1.
#d=2.
#rho0 = lambda x: x*d + c 
#int_rho0 = lambda x : d*x**2/2 +c*x
#int2_rho0 = lambda x : d*x**3/6  +c *x**2/2
#int3_rho0 = lambda x : d*x**4/24 +c *x**3/6
#intp_rho0 = [int_rho0,int2_rho0,int3_rho0] 
#
#rho1 = lambda x: d*(1.-x) + c
#int_rho1 = lambda x : d*(x - x**2/2) + c*x 
#int2_rho1 = lambda x : d*(x**2/2 -x**3/6)  +c* x**2/2
#int3_rho1 = lambda x : d*(x**3/6 -x**4/24) +c* x**3/6
#intp_rho1 = [int_rho1,int2_rho1,int3_rho1] 


c=1.
rho0 = lambda x: np.cos(2*np.pi*x)+c 
int_rho0 = lambda x : np.sin(2*np.pi*x)/(2*np.pi) +c*x
int2_rho0 = lambda x : -np.cos(2*np.pi*x)/(2*np.pi)**2  +c *x**2/2
int3_rho0 = lambda x : -np.sin(2*np.pi*x)/(2*np.pi)**3 +c *x**3/6
intp_rho0 = [int_rho0,int2_rho0,int3_rho0] 

rho1 = lambda x: -np.cos(2*np.pi*x)+c 
int_rho1 = lambda x : -np.sin(2*np.pi*x)/(2*np.pi) + c*x
int2_rho1 = lambda x : np.cos(2*np.pi*x)/(2*np.pi)**2  +c* x**2/2 
int3_rho1 = lambda x : np.sin(2*np.pi*x)/(2*np.pi)**3 +c* x**3/6 
intp_rho1 = [int_rho1,int2_rho1,int3_rho1] 


alpha = 1.0
beta =  0.0

et = Extrapolation1D(alpha,beta,X, rho0,rho1, intp_rho0,intp_rho1)
et.update_weights(maxIter=10000,verbose =True)


increment = .01
n_increment = 10

for i in range(n_increment):
    et.alpha += increment 
    et.beta  += increment 
    et.update_weights(maxIter=10000,verbose =True)














