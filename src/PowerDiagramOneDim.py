""" Power diagram in 1d for cost |x-y|^2 """

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class PowerDiagram1D:
      
    def __init__(self,X,weights=None, L=1):
         """
         Parameters:
         -----------
    
         X: ndarray, Positions of particles (ordered and no repeats)
         weights: ndarray,  weights of Laguerre cells
         L: float,  domain length
         
         """

         self.X = X
         self.weights = weights
         self.L = L
         self.Bounds = np.empty(len(X))
         self.indices = [] 
         self.updated_flag = False
 
    def set_positions(self,X):
         self.X = X.copy()
    
    def set_weights(self,weights):
         self.weights = weights.copy()
         self.updated_flag = False
    
    def update_boundaries(self):
         """ Compute power diagram boundaries via Graham scan
                
         Bounds: ndarray (first and last are 0 and L)
         
         """

         if self.weights is None: self.weights = np.zeros(len(self.X))

         indices = [0, 1]
         u = (self.X**2 - self.weights)/2
         slope = lambda i, j: (u[j] - u[i])/(self.X[j] - self.X[i])
         for i in range(2, len(u)):
            cond = False
            while len(indices)>=2 and not cond:
                slope_r = slope(i, indices[-1])
                slope_l = slope(indices[-1], indices[-2])
                if slope_r <= slope_l:
                    indices.pop()
                else:
                    cond = True
            indices.append(i)
   


 
         Bounds_noconstr =  (u[indices][1:] -u[indices][:-1])/(self.X[indices][1:] - self.X[indices][:-1]) 
         
         mask = Bounds_noconstr<=0
         i0 = np.sum(mask)
         
         mask = Bounds_noconstr>=self.L
         iend = np.sum(mask)
 
         indices = indices[i0:len(indices)-iend]
          
         self.indices = indices
         self.Bounds = np.zeros(len(indices)+1)
         self.Bounds[-1] = self.L
         self.Bounds[1:-1] = Bounds_noconstr[i0:len(Bounds_noconstr)-iend] #(u[indices][1:] -u[indices][:-1])/(self.X[indices][1:] - self.X[indices][:-1])   
        
         self.updated_flag = True         

    
    def compute_energy(self,fun):
         """ Computes power diagram energy sum_i int_{L_i} |x-x_i|^2"""
           
         fplus = (self.Bounds[1:] - self.X[self.indices])**3/3
         fminus =  (self.Bounds[:-1] - self.X[self.indices])**3/3
         energy = np.sum(fplus-fminus)
         return energy  


    def compute_integrals(self, fun):

          """ Computes integral of function rho over laguerre cells 
            
          Parameters:
          ----------
        
          fun: function. Density to integrate  
                 
          """
          if not(self.updated_flag): self.update_boundaries() 
          N = np.size(self.Bounds) -1
          integrals = np.zeros(N)
          for i in range(N):
                integrals[i] = quad(fun,self.Bounds[i],self.Bounds[i+1])[0]

          return integrals
 
    

    def compute_integrals_ipp(self,intp_fun, p=None):
         """ Computes the integral fun(x)*(x-x_i)^p over Laguerres cells, using integration by parts 

         Parameters:
         ---------- 

         intp_rho : list of length p+1, of lambda functions providing [int fun, int int fun , ..]  
         """
         
       
         if p is None: 
             p = len(intp_fun)-1
         else: 
             assert len(intp_fun)>=p+1
        
         if p == 0:
             return intp_fun[0](self.Bounds[1:])-intp_fun[0](self.Bounds[:-1])
         else:
             integrals_p =  ((self.Bounds[1:]-self.X[self.indices])**p*intp_fun[0](self.Bounds[1:]) 
                               - (self.Bounds[:-1]-self.X[self.indices])**p*intp_fun[0](self.Bounds[:-1]))
             integrals = -p*self.compute_integrals_ipp(intp_fun[1:],p=p-1) + integrals_p
             return integrals
         

    def compute_integrals_gradient(self, fun):
 
          """ Computes gradient with respect to weights of 
                           G: w -->  integrals of function rho over lagurre cells   
                 
          """
          if not(self.updated_flag): self.update_boundaries() 
          
          # Force lower bound on density 
          

            
          Xact = self.X[self.indices].copy()      
        


          feval = fun(self.Bounds[1:-1]) 
          fmin = np.min(feval)
          feval = feval*(1-1e-2)+1e-2 #- fmin +   np.max((1e-2,fmin))


          #(np.maximum(fun(self.Bounds[1:-1])-fmin,0)+fmin)
          vect = 0.5 * feval/np.abs(Xact[1:] - Xact[:-1])
          vect0 = vect.copy()
          vect0[0] = 0
          costhess = diags(vect,-1) + diags(vect0,1)
          vect1 = np.array(np.sum(costhess,axis=1)).flatten()
          
          # First row is zero apart from first element
          vect1[0] = -1
          costhess -= diags(vect1)      
            
          return -costhess


