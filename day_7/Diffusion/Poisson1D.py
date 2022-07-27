#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator
"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#%matplotlib qt
plt.close()

Ns = np.array([8,16,31,64])
convergence = []; e_old = 1
for N in Ns:
    "Number of points"
    Dx = 1/N
    x = np.linspace(0,1,N+1)
    
    "System matrix and RHS term"
    A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    A[:,0] = np.concatenate(([1], np.zeros(N))); A[0,1] = 0
    
    order = 0
    if(order == 2): ## Second Order
        A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2, 3/2]))
    elif(order == 1): ## First Order Order
        A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
    else:
        x = np.linspace(0,1+Dx,N+2)
        A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
        A[:,0] = np.concatenate(([1], np.zeros(N+1))); A[0,1] = 0
        A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1/2,0,1/2]))
        
    
    F = 2*(2*x**2+5*x-2)*np.exp(x)
    F[0] = 0; F[-1] = 0
    
    "Solution of the linear system AU=F"
    U = np.linalg.solve(A,F)
    u = U # np.concatenate(([0],U,[0]))
    ua = 2*x*(3-2*x)*np.exp(x)
    
    "Plotting solution"
    plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
    plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
    plt.legend(fontsize=12,loc='upper left')
    plt.grid()
    plt.axis([0, 1,0, 6])
    plt.xlabel("x",fontsize=16)
    plt.ylabel("u",fontsize=16)
    
    "Compute error"
    error = np.max(np.abs(u-ua))
    convergence.append(error/e_old)
    e_old = error
    print("Linf error u: %g\n" % error)
print(convergence)





