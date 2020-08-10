""" Solves Heat equation using a Moreau-Yosida regularization of the entropy with respect to the particle masses """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, Bounds
from semidiscrete_tools import * 
from scipy.integrate import quad
import numpy.random as npr 

L = 1.

# Density (mass is normalized to 1)
rho0_temp = lambda x: np.sin(2*np.pi*x)**2+.1


mass = quad(rho0_temp,0,L)
rho0 = lambda x: rho0_temp(x)/mass[0]

# Number of particles
N = 50

# Position of particles
X = np.linspace(L/(2*N),L - L/(2*N),N)

# Initial guess weights Laguerre cells
weights = np.ones(N)*10

# Masses of particles and initial tessellation (Voronoi)

Bounds0 = laguerre(X,weights,L)
rho0masses = integralrho(Bounds0,rho0)



# Time integration 
###########################################################

tau = 0.01
epsilon = 0.05 


Ntot = 100

bounds=Bounds(1e-5, np.inf)


Xtot = np.empty((Ntot+1,N))
Xtot[0,:] = X

for k in range(Ntot):
     
    cost = lambda w : evalcostEntropy(w,rho0masses,X, epsilon,L)
    costgrad = lambda w : evalcostgradEntropy(w, rho0masses,X, epsilon,L)
    costhess = lambda w : evalcosthessEntropy(w,rho0masses,X, epsilon,L)

    res = minimize(cost ,weights, method= 'trust-constr', jac=costgrad, hess = costhess,  options={'xtol': 1e-08,'disp': True},bounds=bounds)

    cells = laguerre(X,res.x,L)
    cellsbary = .5*(cells[1:] + cells[:-1])
    X = X - (X-cellsbary)/epsilon*tau  
    Xtot[k+1,:] = X
    print("Iteration {}".format(k))



# Minimization with constraint of no vanishing cells
###############################################################
#rhoLeb = lambda x: 1.
#constraint = lambda w : novanishconstraint(w,X,L) 
#constraintgrad = lambda w: evalcosthess(w,X,rhoLeb,L)
#nonlinear_constraint = NonlinearConstraint(constraint, 0 ,np.inf, jac=constraintgrad)
#res = minimize(cost ,weights, method='trust-constr', constraints = [nonlinear_constraint], jac=costgrad,hess =costhess, options={'disp': True})


# Unconstrained minimization
###############################################################
#res = minimize(cost ,weights, method='Newton-CG', jac=costgrad,hess =costhess, options={'xtol': 1e-08,'disp': True})

#cells = laguerre(X,res.x,L)
#cellsbary = .5*(cells[1:] + cells[:-1])


#cellsdens = masses*1./(cells[1:]-cells[:-1])

#fig,ax = plt.subplots(1)
#ax.plot(cellsbary,cellsdens,'r')
#ax.plot(cellsbary,rho(cellsbary),'b--')
#fig.show()
