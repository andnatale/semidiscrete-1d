""" Semi-discrete tools for Moreau Yosida regularization of Power energy
                          Energy: E(rho) = rho^m/(m-1)
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

class SemiDiscreteSolverPowers:
    """ One dimensional solver for Moreau Yosida regularization of power energies """
    def __init__(self, X, masses, L, epsilon, power):
         """
         :param X: Positions of particles (ordered) 
         :param L: Dimension of domain 
         :param epsilon: Moreau-Yosida regularization parameter
         :param power: E(rho) = rho^power/(power-1)    
         """          
 
         self.X = X
         self.masses = masses 
         self.L = L
         self.epsilon = epsilon
         self.power = power
         self.r = power/(power-1)
 
         # Weights 
         self.weights = np.ones(X.shape)

         # Cell boundaries
         self.Boundlow = np.zeros(X.shape)
         self.Boundup = np.zeros(X.shape)



    def updateLaguerre(self, weights):
         """ Updates position of Laguerre cells boundaries """

         N = np.size(self.X)
         Bounds = np.zeros(N+1)
         Bounds[-1] = self.L
         Bounds[1:-1] =  0.5*(self.X[1:]+self.X[:-1] - (weights[1:] - weights[:-1])/(self.X[1:]-self.X[:-1]))

         Bounds[Bounds>=self.L] = self.L
         Bounds[Bounds<=0] = 0 
        
         self.Boundlow = Bounds[:-1].copy()
         self.Boundup = Bounds[1:].copy()

         pos_flag = weights >0

         self.Boundup[pos_flag] = np.minimum(self.Boundup[pos_flag],  self.X[pos_flag]+ np.sqrt(weights[pos_flag]))
         self.Boundlow[pos_flag] = np.maximum(self.Boundlow[pos_flag],  self.X[pos_flag]- np.sqrt(weights[pos_flag]))
         
         return np.min(self.Boundup-self.Boundlow)<=1e-10 or np.min(self.Boundlow[1:]-self.Boundlow[:-1])<=1e-10 or np.min(self.Boundup[1:]-self.Boundup[:-1])<=1e-10 
    def quadratureCost(self):
         """ Computes integral of cost to minimize: -m *weights/(2epsilon) +((-|x-X|^2+weights + mu)/(2 epsilon r))^r over all cells """
  
         L = self.Boundlow
         U = self.Boundup
         X = self.X
         w = self.weights
         c = self.r *2*self.epsilon
        
         if self.power == 2:
               integrals = ((-L**5/5 + L**4*X +(2/3)*L**3*(w-3*X**2) +2 * L**2*X*(X**2-w) -L *(w-X**2)**2 + U**5/5 -U**4*X -(2/3)*U**3*(w-3*X**2) +2*U**2*X*(w -X**2) + U*(w-X**2)**2)/c**2) 
  
         else: 
               raise ValueError("Powers different from 2 not implemented!")

         return - np.dot(self.masses,self.weights)/(2*self.epsilon) + np.sum(integrals)
 
    def quadratureDerCost(self):
         """ Computes integral of  derivative wrt weights  of cost to minimize: - m/(2*epsilon) + ((-|x-X|^2+weights + mu)/(2 epsilon r))^r over cells """
         
         L = self.Boundlow
         U = self.Boundup
         X = self.X
         w = self.weights
         c = self.r *2*self.epsilon
        
         if self.power == 2:
               integrals = (1/3)* (L - U) *(L**2 + U**2 - 3* w + L* (U - 3* X) - 3* U* X + 3* X**2)/(8*self.epsilon**2)
         else: 
               raise ValueError("Powers different from 2 not implemented!")

         return -self.masses/(2*self.epsilon)+ integrals
 
    def quadratureHessCost(self):
         """ Computes integral of Hessian wrt weights  of cost to minimize: - m/(2*epsilon) + ((-|x-X|^2+weights + mu)/(2 epsilon r))^r over cells """
         
         L = self.Boundlow
         U = self.Boundup
         X = self.X
         w = self.weights
         c = self.r *2*self.epsilon

         if self.power == 2:
                   
               Hess = np.diag((U-L)/(8*self.epsilon**2)) 
               updiag = (-(U[:-1]-X[:-1])**2 + w[:-1])/ (8*self.epsilon**2* np.abs(X[:-1]-X[1:])) 
               #lowdiag = (-(L[:-1]-X[:-1])**2 + w[:-1])/ (8*self.epsilon**2* np.abs(X[:-1]-X[1:])) 
               maindiag = np.zeros(updiag.size+1)
               maindiag[:-1] = -updiag
               maindiag[1:] = maindiag[1:] -updiag
               Hess = Hess -( np.diag(updiag ,1) + np.diag(updiag,-1) + np.diag(maindiag))
 
         else: 
               raise ValueError("Powers different from 2 not implemented!")
 
         return Hess


    def updateWeights(self, tol = 1e-6,maxIter = 1e3, verbose = False):
         self.updateLaguerre(self.weights)
         fun  =  self.quadratureDerCost() 
         Hess = self.quadratureHessCost()
         i = 0  
         tau = 1.
         error = 1.
         
         while error> tol and i< maxIter:
               flag = True
               j = 0
               while flag:             
                   weights = self.weights  - tau* np.linalg.solve(Hess, fun)
                   flag = self.updateLaguerre(weights)
                   tau = .9*tau 
                   j +=1 
                   if j>= maxIter: raise ValueError("Negative area cells.. maximum iteration exceeded")
               
               tau = 1.
               self.weights = weights.copy()  
               fun = self.quadratureDerCost()
               Hess = self.quadratureHessCost()
               error = np.max(np.abs(fun))
               i +=1
               if verbose:
                    print('Iteration: ',i,' Error: ', error, 'Cost: ', self.quadratureCost())

         if i == maxIter:
                print('Warning!! Maximum iteration exceeded')
         else: 
                if verbose: print('Optimization success!')

    def computeBarycenters(self):
 
         L = self.Boundlow
         U = self.Boundup
         X = self.X
         w = self.weights
 
         if self.power ==2:
              integrals = L**4/4 - U**4/4 - (L**2* w)/2 + (U**2* w)/2 - (2* L**3 *X)/3 + (2* U**3* X)/3 + (L**2* X**2)/2 - (U**2* X**2)/2
              barycenters = 1/(4 *self.epsilon)*integrals/self.masses
         else: 
              raise ValueError("Powers different from 2 not implemented!")
         
         return barycenters





 
