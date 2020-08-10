import numpy as np
import matplotlib.pyplot as plt
import semidiscrete_tools_powers as sd       

L = 1.
N = 200
delta = L- 1/N
X = np.linspace(L/2-delta/2,L/2 +delta/2,N)
#X = X**2
masses =np.ones(X.size)/N
epsilon = 1./N
power = 2

solver = sd.SemiDiscreteSolverPowers(X,masses,L,epsilon,power)
solver.updateWeights(verbose = True)

# Time step
tau = epsilon/100

Nt = int(2./tau)

Xevol = np.zeros((Nt+1,X.size))

Xevol[0,:] = X
Vevol = Xevol.copy()
# Initial velocity for second order model
Vevol[0,:] =2*X*(1. - X)#.1*(X<L/2) #5*(-(.5 - X)**2+.25)

# First order model
first = False



for k in range(Nt):
    B = solver.computeBarycenters()
   
    if first:
         Xevol[k+1,:] = B + np.exp(-tau/epsilon)*(Xevol[k,:] - B)
    else: 
         Xevol[k+1,:] = B + np.cos(tau/np.sqrt(epsilon)) * (Xevol[k,:]-B) + np.sin(tau/np.sqrt(epsilon))*Vevol[k,:]*np.sqrt(epsilon)
         Vevol[k+1,:] = -np.sin(tau/np.sqrt(epsilon)) * (Xevol[k,:]-B)/np.sqrt(epsilon) + np.cos(tau/np.sqrt(epsilon))*Vevol[k,:]


    solver.X = Xevol[k+1,:]
    solver.updateWeights()
    print('Time iteration: ',k)




