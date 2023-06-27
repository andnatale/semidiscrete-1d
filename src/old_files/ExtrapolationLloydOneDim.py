""" Extrapolation in 1d for cost |x-y|^2 using Lloyd's algorithm """

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from PowerDiagramOneDim import *

class ExtrapolationLloyd1D():
    """ Extrapolation problem with fixed masses and moving arrival positions"""

    def __init__(self,alpha,beta,X, masses, rho0,rho1, intp_rho0,intp_rho1, L=1.):
    
         """
         Parameters:
         -----------
          
         alpha,beta: parameters of extrapolation minimization  alpha W^2(rho,rho_1) - beta W^2(rho,rho^0)
         X: ndarray, Initial positions of particles of extrapolated measure (ordered and no repeats)
         masses: ndarray, Masses of particles
         weights: ndarray,  weights of Laguerre cells for transport from rho to rho^0
         L: float,  domain length
         rho0,rho1 : functions
         intprho0, intprho1: list of p integrals of rho0 and rho1, ex:  [int rho, int int rho , ..]        
         """
      
         self.alpha = alpha
         self.beta = beta
         self.X = X
         self.L = L
         self.ot0 = OptimalTransport1D(X,masses,rho0,intp_rho = intp_rho0, L=L)
         self.ot1 = OptimalTransport1D(X,masses,rho1,intp_rho = intp_rho1, L=L)
         self.masses = masses  
         self.rho0 =rho0
         self.intp_rho0 = intp_rho0
         self.rho1 = rho1
         self.intp_rho1 = intp_rho1
  
     
    def compute_extrapolation_cost(self):

         cost0 = self.ot0.compute_ot_cost_ipp()
         cost1 = self.ot1.compute_ot_cost_ipp()
         return self.alpha*cost1 - self.beta* cost0


    def update_positions(self, tol = 1e-6,maxIter = 5e2, verbose = False):
 
         i = 0
         error = 1.
         self.ot0.update_weights(tol=tol,maxIter=maxIter,verbose=verbose)
         self.ot1.update_weights(tol=tol,maxIter=maxIter,verbose=verbose)
         
         tau =.5  
         cost = 0
         cost_old = 0 
         while error>tol and i < maxIter: 
          
             bary0 = self.ot0.compute_integrals_ipp(self.intp_rho0,p=1)/self.masses + self.X
             bary1 = self.ot1.compute_integrals_ipp(self.intp_rho1,p=1)/self.masses + self.X
             
             oldX = self.X.copy()
             while cost >= cost_old:
                  newX = np.sort(oldX*(1-tau)+tau* (self.alpha*bary1 -self.beta*bary0)/(self.alpha-self.beta))
                  error = np.linalg.norm(newX - self.X)    
                  self.X = newX.copy()        
                  self.ot0.set_positions(self.X)
                  self.ot0.set_weights(0*self.ot0.weights)
                  self.ot1.set_positions(self.X) 
                  self.ot1.set_weights(0*self.ot1.weights)
             
                  self.ot0.update_weights(tol=tol,verbose=verbose)
                  self.ot1.update_weights(tol=tol,verbose=verbose)

                  cost = self.compute_extrapolation_cost() 
                  tau = tau/2
                  print(tau)       
             cost_old = cost
             i+=1
          
             if verbose: print("Lloyd step: {}, cost: {}, error: {}".format(i,cost,error))     
               

         if i< maxIter and verbose: print("Optimization success!")



N = 1000
L = 1.
masses = np.ones(N)
#masses[50:86]=1e-6#0.001
#X = np.sort(np.random.rand(N))
#X = (X + np.linspace(0,1,N+1)[:-1])/2
X = np.linspace(1/2/N,1-1/2/N,N)

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



mass = int_rho0(L)-int_rho0(0)
masses = masses/np.sum(masses)*mass
alpha = 1.5#000000001
beta =  .5#000000001


et = ExtrapolationLloyd1D(alpha,beta,X,masses, rho0,rho1, intp_rho0,intp_rho1,L)
et.update_positions(tol=1e-6,maxIter=10000,verbose =True)











#ot = OptimalTransport1D(X,masses,rho0,intp_rho= intp_rho0, L=L)#intp_rho = intp_rho0,L=L)
#ot.update_weights(maxIter=10000,verbose =True)



