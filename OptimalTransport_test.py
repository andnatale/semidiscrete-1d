"""Test for Optimal Transport sovler"""

import numpy as np
import matplotlib.pyplot as plt
from src.OptimalTransportOneDim import OptimalTransport1D


N = 100 
L = 1.
masses = np.ones(N)
#masses[50:86]=1e-6#0.001
masses = masses/np.sum(masses)
#X = np.sort(np.random.rand(N))
#X = (X + np.linspace(0,1,N+1)[:-1])/2
X = np.linspace(1/2/N,1-1/2/N,N)

# TEST OT
c= 0.
rho0 = lambda x: np.maximum(x-.5,0) +c
int_rho0 = lambda x : rho0(x)**2/2 +c*x
int2_rho0 = lambda x : rho0(x)**3/6 + c*x**2/2
int3_rho0 = lambda x : rho0(x)**4/24+ c*x**3/6
intp_rho0 = [int_rho0,int2_rho0,int3_rho0]

mass = int_rho0(L)-int_rho0(0)
masses = masses*mass

ot = OptimalTransport1D(X,masses,rho0,intp_rho= intp_rho0, L=L)#intp_rho = intp_rho0,L=L)
ot.update_weights(maxIter=10000,verbose =True)





