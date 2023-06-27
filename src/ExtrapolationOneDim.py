""" Extrapolation in 1d for cost |x-y|^2 """

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .PowerDiagramOneDim import *

class Extrapolation1D():
    """ Extrapolation problem with fixed arrival positions and variable masses"""

    def __init__(self,alpha,beta,X, rho0,rho1, intp_rho0,intp_rho1,weights = None, L=1):
    
         """
         Parameters:
         -----------
          
         alpha,beta: parameters of extrapolation minimization  alpha W^2(rho,rho_1) - beta W^2(rho,rho^0)
         X: ndarray, Positions of particles of extrapolated measure (ordered and no repeats)
         weights: ndarray,  weights of Laguerre cells for transport from rho to rho^0
         L: float,  domain length
         rho0,rho1 : functions
         intprho0, intprho1: list of p integrals of rho0 and rho1, ex:  [int rho, int int rho , ..]        
         """
      
         self.alpha = alpha
         self.beta = beta
         self.X = X.copy()
         if weights is None: self.weights = np.zeros(len(X))
         else: self.weights = weights.copy()
         self.L = L
         self.pd0 = PowerDiagram1D(X, weights = self.weights, L=L)
         self.pd1 = PowerDiagram1D(X, weights = self.beta/self.alpha* self.weights, L=L)
        
         self.rho0 =rho0
         self.intp_rho0 = intp_rho0
         self.rho1 = rho1
         self.intp_rho1 = intp_rho1
  
    def set_weights(self,weights):
         self.weights = weights
         self.pd0.set_weights(weights)
         self.pd1.set_weights(self.beta/self.alpha*weights)
    
           
     
    def compute_extrapolation_cost(self):

         cost0 = np.sum(self.pd0.compute_integrals_ipp(self.intp_rho0,p=2)
                                -self.pd0.compute_integrals_ipp(self.intp_rho0,p=0)*self.pd0.weights[self.pd0.indices])
         cost1 = np.sum(self.pd1.compute_integrals_ipp(self.intp_rho1,p=2)
                                -self.pd1.compute_integrals_ipp(self.intp_rho1,p=0)*self.pd1.weights[self.pd1.indices])
        
         return self.alpha*cost1 - self.beta* cost0


    def update_weights(self, tol = 1e-6,maxIter = 5e2, verbose = False):
       
         self.pd0.update_boundaries()
         self.pd1.update_boundaries()

         i = 0
         error = 1.
         tau = 1e-1
         c_armijo = 0#1e-4 
 
         F = np.zeros(len(self.X))
         F[self.pd0.indices] +=  self.pd0.compute_integrals_ipp(self.intp_rho0,p=0)
         F[self.pd1.indices] -=  self.pd1.compute_integrals_ipp(self.intp_rho1,p=0)

         cost_old = self.compute_extrapolation_cost()
 
         while error>tol and i < maxIter:

                Hess = - self.beta/self.alpha* self.pd1.compute_integrals_gradient(self.rho1) +  self.pd0.compute_integrals_gradient(self.rho0)
                #print(self.indices)
                #if tau<1e-9: theta=1. 

                theta = 0. 
                deltaw = - theta*F
                deltaw  -= (1-theta)*spsolve(Hess,F)

                weights_old = self.weights.copy()
                k=0

                # Linesearch
                while True:
                      self.set_weights( weights_old +tau*deltaw)
                      self.pd0.update_boundaries()
                      self.pd1.update_boundaries()
                      #cost = self.compute_ot_cost()


                      cost = self.compute_extrapolation_cost()

                      if (cost <= cost_old
                           and len(self.pd0.indices)==len(self.X)
                           and len(self.pd1.indices) == len(self.X))  or tau<1e-10: break
                      #print(np.dot(deltaw,F))
                      #if cost <=cost_old + c_armijo*tau*np.dot(deltaw,F) : break    
                  
                      else:
                            k += 1
                            tau = tau*.5

                            #print(deltaw)
                #if i>200: tau = np.min((1., tau*1.01))

                cost_old = cost

                i+=1 
                F = np.zeros(len(self.X))
                F[self.pd0.indices] += self.pd0.compute_integrals_ipp(self.intp_rho0,p=0)
                F[self.pd1.indices] -= self.pd1.compute_integrals_ipp(self.intp_rho1,p=0)

                error = np.linalg.norm(F) 
            
                if verbose: print("Newton step: {}, cost: {}, tau: {}, error: {}, active particles rho0: {}".format(i,cost,tau,error,len(self.pd0.indices)))     
                tau = np.min((tau*1.1,1.))

         if i< maxIter and verbose: print("Optimization success!")





