""" Extrapolation in 1d for cost |x-y|^2 """

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from PowerDiagramOneDim import *

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
 
         F = np.zeros(len(X))
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
                           and len(self.pd0.indices)==len(X)
                           and len(self.pd1.indices) == len(X))  or tau<1e-10: break
                      #print(np.dot(deltaw,F))
                      #if cost <=cost_old + c_armijo*tau*np.dot(deltaw,F) : break    
                  
                      else:
                            k += 1
                            tau = tau*.5

                            #print(deltaw)
                #if i>200: tau = np.min((1., tau*1.01))

                cost_old = cost

                i+=1 
                F = np.zeros(len(X))
                F[self.pd0.indices] += self.pd0.compute_integrals_ipp(self.intp_rho0,p=0)
                F[self.pd1.indices] -= self.pd1.compute_integrals_ipp(self.intp_rho1,p=0)

                error = np.linalg.norm(F) 
            
                if verbose: print("Newton step: {}, cost: {}, tau: {}, error: {}, active particles rho0: {}".format(i,cost,tau,error,len(self.pd0.indices)))     
                tau = np.min((tau*1.1,1.))

         if i< maxIter and verbose: print("Optimization success!")



N = 10000 
L = 1.
masses = np.ones(N)
#masses[50:86]=1e-6#0.001
masses = masses/np.sum(masses)
#X = np.sort(np.random.rand(N))
#X = (X + np.linspace(0,1,N+1)[:-1])/2
X = np.linspace(1/2/N,1-1/2/N,N)

# TEST OT
#c= 0.1
#rho0 = lambda x: np.maximum(x-.5,0) +c
#int_rho0 = lambda x : rho0(x)**2/2 +c*x
#int2_rho0 = lambda x : rho0(x)**3/6 + c*x**2/2
#int3_rho0 = lambda x : rho0(x)**4/24+ c*x**3/6
#intp_rho0 = [int_rho0,int2_rho0,int3_rho0]
#
#mass = int_rho0(L)-int_rho0(0)
#masses = masses*mass
#
#ot = OptimalTransport1D(X,masses,rho0,intp_rho= intp_rho0, L=L)#intp_rho = intp_rho0,L=L)
#ot.update_weights(maxIter=10000,verbose =True)

# TEST EXTRAPOLATION
c=1.
d=2.
rho0 = lambda x: x*d + c 
int_rho0 = lambda x : d*x**2/2 +c*x
int2_rho0 = lambda x : d*x**3/6  +c *x**2/2
int3_rho0 = lambda x : d*x**4/24 +c *x**3/6
intp_rho0 = [int_rho0,int2_rho0,int3_rho0] 

rho1 = lambda x: d*(1.-x) + c
int_rho1 = lambda x : d*(x - x**2/2) + c*x 
int2_rho1 = lambda x : d*(x**2/2 -x**3/6)  +c* x**2/2
int3_rho1 = lambda x : d*(x**3/6 -x**4/24) +c* x**3/6
intp_rho1 = [int_rho1,int2_rho1,int3_rho1] 


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


alpha = 1.0#000000001
beta =  0.0#000000001


et = Extrapolation1D(alpha,beta,X, rho0,rho1, intp_rho0,intp_rho1)
et.update_weights(maxIter=10000,verbose =True)



















