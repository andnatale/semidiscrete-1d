""" Solves semi-discrete optimal transport problem between a given density and an empirical distribution """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from semidiscrete_tools import * 
from scipy.integrate import quad
import numpy.random as npr 

L = 1.

# Density (mass is normalized to 1)
rho0 = lambda x: np.sin(2*np.pi*x)**2
mass = quad(rho0,0,L)
rho = lambda x: rho0(x)/mass[0]

# Number of particles
N = 200

# Position of particles
X = np.linspace(0.1,.9,N)

# Initial guess weights Laguerre cells
weights = np.zeros(N)

# Masses of particles
masses = np.ones(N)*1./N

cost = lambda w : evalcost(w, masses,X, rho)
costgrad = lambda w : evalcostgrad(w, masses,X, rho)
costhess = lambda w : evalcosthess(w,X, rho,L)


# Minimization with constraint of no vanishing cells
###############################################################
#rhoLeb = lambda x: 1.
#constraint = lambda w : novanishconstraint(w,X,L) 
#constraintgrad = lambda w: evalcosthess(w,X,rhoLeb,L)
#nonlinear_constraint = NonlinearConstraint(constraint, 0 ,np.inf, jac=constraintgrad)
#res = minimize(cost ,weights, method='trust-constr', constraints = [nonlinear_constraint], jac=costgrad,hess =costhess, options={'disp': True})


# Unconstrained minimization
###############################################################
res = minimize(cost ,weights, method='Newton-CG', jac=costgrad,hess =costhess, options={'xtol': 1e-08,'disp': True})

cells = laguerre(X,res.x,L)
cellsbary = .5*(cells[1:] + cells[:-1])
cellsdens = masses*1./(cells[1:]-cells[:-1])

fig,ax = plt.subplots(1)
ax.plot(cellsbary,cellsdens,'r')
ax.plot(cellsbary,rho(cellsbary),'b--')
fig.show()
