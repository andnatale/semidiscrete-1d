""" Optimal  in 1d for cost |x-y|^2 """

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .PowerDiagramOneDim import PowerDiagram1D

class OptimalTransport1D(PowerDiagram1D):
    def __init__(self,X,masses,rho,intp_rho=None, L=1):
          super().__init__(X,L=L) 
          self.masses = masses
          self.rho = rho
          self.intp_rho = intp_rho #list of integrals of rho
     
    def compute_ot_cost(self):

          N = np.size(self.Bounds) -1
          integrals = np.zeros(N)
          for i in range(N):
                fun = lambda x: ( (x-self.X[self.indices][i])**2 - self.weights[self.indices][i])*self.rho(x)
                integrals[i] = quad(fun,self.Bounds[i],self.Bounds[i+1])[0]
                                
          return np.sum(integrals)+np.sum(self.masses*self.weights)
          
    
    def compute_ot_cost_ipp(self):     
         
          cost = np.sum(self.masses*self.weights) + np.sum(self.compute_integrals_ipp(self.intp_rho,p=2)
                                - self.compute_integrals_ipp(self.intp_rho,p=0)*self.weights[self.indices])
          return cost

    def update_weights(self,tol = 1e-6,maxIter = 5e2, verbose = False):
          """
          Computes optimal weights with damped Newton 
          
          """
          # Armijo parameter
          alphaA = 0.01

          self.update_boundaries()
          
          i = 0
          tau = .5
          F = -self.masses.copy()
          if self.intp_rho is None:
                F[self.indices] += self.compute_integrals(self.rho)
          else: 
                F[self.indices] += self.compute_integrals_ipp(self.intp_rho,p=0) 
          #F[self.indices] += self.compute_integrals(self.rho)

          error = np.linalg.norm(F) 
          #cost_old =  self.compute_ot_cost() 
          if self.intp_rho is None :
                 cost_old = self.compute_ot_cost()
          else: 
                 cost_old = self.compute_ot_cost_ipp()
     
          while error>tol and i<maxIter:
                        
                Hess = self.compute_integrals_gradient(self.rho)   
                #print(self.indices)
                #if tau<1e-9: theta=1. 
                
                theta = 0.
                deltaw = -theta*F
                deltaw[self.indices]  -= (1-theta)*spsolve(Hess,F[self.indices])
                
                weights_old = self.weights.copy()
                k=0
               
                # Linesearch
                while True:
                      self.weights = weights_old +tau*deltaw
                      self.update_boundaries()
                      #cost = self.compute_ot_cost()

                      if self.intp_rho is None :
                            cost = self.compute_ot_cost()
                      else: 
                            cost = (np.sum(self.masses*self.weights)
                                    +np.sum(self.compute_integrals_ipp(self.intp_rho,p=2)
                                    -self.compute_integrals_ipp(self.intp_rho,p=0)*self.weights[self.indices]))
                      
                      if (cost >= cost_old + tau*alphaA*np.dot(F,deltaw)
                           and len(self.indices)==len(self.X)) or tau<1e-10: break
                                           
                      else: 
                            k += 1
                            tau = tau*.8 
     
                            #print(deltaw)
                #if i>200: tau = np.min((1., tau*1.01))
               
                cost_old = cost
          
                #self.weights = weights_new.copy()    
                #self.update_boundaries()
                #print(cost,tau)
                i+=1
                F = -self.masses.copy()
                if self.intp_rho is None:
                      F[self.indices] += self.compute_integrals(self.rho)
                else:
                      F[self.indices] += self.compute_integrals_ipp(self.intp_rho,p=0)
                #F[self.indices] += self.compute_integrals(self.rho) 
                error = np.linalg.norm(F) 
            
                if verbose: print("Newton step: {}, cost: {}, tau: {}, error: {}, active particles: {}".format(i,cost,tau,error,len(self.indices)))     
                tau = np.min((tau*1.1,1.))

          if i< maxIter and verbose: print("Optimization success!")


 










