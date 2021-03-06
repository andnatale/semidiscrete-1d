""" Semi-discrete tools for cost |x-y|^2 """

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def laguerre(X,weights,L):
    """ Gives position of Laguerre cells boundaries

    Parameters:
    -----------
    
    X: ndarray
       Positions of particles (ordered)
    
    weights: ndarray
       weights of Laguerre cells
    
    L: float
       domain length
    
    Returns:
    --------
    
    Bounds: ndarray
       Position of cells boundaries (first and last are 0 and L)

    """
    N = np.size(X)
    Bounds = np.zeros(N+1)
    Bounds[-1] = L
    Bounds[1:-1] =  0.5*(X[1:]+X[:-1] - (weights[1:] - weights[:-1])/(X[1:]-X[:-1]))
  
    Bounds[Bounds>=L] = L
    Bounds[Bounds<=0] = 0 
    
    return Bounds 


def integralrho(Bounds,rho):
   """ Integrals of density on Laguerre cells
    
   Parameters:
   ----------
   
   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  
 
   Returns:
   --------
   
   Integrals: ndarray
      Integrals of rho on cells
   
   """
   N = np.size(Bounds) -1
   Integrals = np.zeros(N)
   for i in range(N):
       Integrals[i] = quad(rho,Bounds[i],Bounds[i+1])[0]
   return Integrals



def integralrhoCost(X,Bounds,rho):
   """ Integrals of density times |x-X_i|^2  on Laguerre cells of particles located at X_i
    
   Parameters:
   ----------
   X: ndarray
      Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  
 
   Returns:
   --------
   
   Integrals: ndarray
      Integrals of rho(x)*|x-X_i|^2 on i-th cell
   
   """
   N = np.size(Bounds) -1
   Integrals = np.zeros(N)
   for i in range(N):
       fun = lambda x: rho(x)*(x-X[i])**2
       Integrals[i] = quad(fun,Bounds[i],Bounds[i+1])[0]
   return Integrals

 



def evalcost(weights, masses,X, rho,L=1.):
   """ Computes the cost to minimize 
    
   Parameters:
   ----------
   masses: ndarray
       Masses of particles

   weights: ndarray
       weights of Laguerre cells

   X: ndarray
       Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  
 
   Returns:
   --------
   
   cost: float 
   
   costgrad: ndarray
      Gradient of cost wrt weights
   """
   Bounds = laguerre(X,weights,L) 
   Irc  =  integralrhoCost(X,Bounds,rho)
   Ir = integralrho(Bounds,rho)
   cost = -(np.dot(masses,weights) + np.sum(Irc) -  np.dot(Ir,weights))    

   return cost 


def evalcostgrad(weights,masses,X, rho,L=1.):
   """ Computes the cost gradient with respect to the weights
    
   Parameters:
   ----------
   masses: ndarray
       Masses of particles

   weights: ndarray
       weights of Laguerre cells

   X: ndarray
       Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  
 
   Returns:
   --------
   
   cost: float 
   
   costgrad: ndarray
      Gradient of cost wrt weights
   """
   Bounds = laguerre(X,weights,L)
   Irc  =  integralrhoCost(X,Bounds,rho)
   Ir = integralrho(Bounds,rho)
   costgrad =  -masses + Ir 
      
 
   return  costgrad 

def evalcosthess(weights,X, rho,L=1.):
   """ Computes the cost hessian with respect to the weights
    
   Parameters:
   ----------
   masses: ndarray
       Masses of particles

   weights: ndarray
       weights of Laguerre cells

   X: ndarray
       Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  
 
   Returns:
   --------
    
   costhess: ndarray
      Hessian of cost wrt weights
   """
   Bounds = laguerre(X,weights,L)
   N = np.size(weights)
   vect = 0.5*rho(Bounds[1:-1])/np.abs(X[1:] - X[:-1])
   costhess = np.diag(vect,-1) + np.diag(vect,1)
   costhess -= np.diag(np.sum(costhess,axis=1)) 

   
   return -costhess


def novanishconstraint(weights,X,L):
   Bounds = laguerre(X,weights,L)
   return Bounds[1:] - Bounds[:-1]


#############################################################################################
# Functions for unbalanced gradient flows ###################################################
#############################################################################################






def evalcostEntropy(weights, rho0masses ,X, epsilon, L=1.):
   """ Computes the cost to minimize: Moreau Yosida regularization with fixed density

                          W^2_2( rho, M_i/|P_i| delta_X_i  )/2epsilon + U(rho0/M)   
    
   Parameters:
   ----------
   rho0masses: ndarray
       integrals of initial density rho0 on reference domain decomposition 

   weights: ndarray
       weights of Laguerre cells

   X: ndarray
       Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  

   epsilon: float
       Moreau-Yosida regularization
 
   Returns:
   --------
   
   cost: float 
   
   costgrad: ndarray
      Gradient of cost wrt weights
   """
   
   Leb = lambda x : 1.
   
   Bounds = laguerre(X,weights,L) 
   Irc  =  integralrhoCost(X,Bounds,Leb)
   Ir = integralrho(Bounds,Leb)
   cost = -(np.dot(rho0masses,np.log(weights/(2*epsilon))) + (np.sum(Irc) -  np.dot(Ir,weights))/(2*epsilon))    

   return cost 


def evalcostgradEntropy(weights, rho0masses,X, epsilon, L=1.):
   """ Computes the cost gradient with respect to the weights
    
   Parameters:
   ----------
   rho0masses: ndarray
       integrals of initial density rho0 on reference domain decomposition 

   weights: ndarray
       weights of Laguerre cells

   X: ndarray
       Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   rho: function
      Density to integrate  
 
   Returns:
   --------
   
   cost: float 
   
   costgrad: ndarray
      Gradient of cost wrt weights
   """

   Leb = lambda x : 1.
   
   Bounds = laguerre(X,weights,L)
   Irc  =  integralrhoCost(X,Bounds,Leb)
   Ir = integralrho(Bounds,Leb)
   costgrad =  -rho0masses/weights + Ir/(2*epsilon) 
      
 
   return  costgrad 

def evalcosthessEntropy(weights, rho0masses, X, epsilon, L=1.):
   """ Computes the cost hessian with respect to the weights
    
   Parameters:
   ----------
   rho0masses: ndarray
       integrals of initial density rho0 on reference domain decomposition 

   weights: ndarray
       weights of Laguerre cells

   X: ndarray
       Positions of particles (ordered)

   Bounds: ndarray
      Position of cells boundaries (first and last are 0 and L)
     
   Returns:
   --------
    
   costhess: ndarray
      Hessian of cost wrt weights
   """ 
   Leb = lambda x : 1
   costhess =  evalcosthess(weights,X,Leb,L)/(2*epsilon) + np.diag(rho0masses/(weights**2))
   
   return costhess





