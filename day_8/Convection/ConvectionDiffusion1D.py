#!/usr/bin/env python
"""
Solution of a 1D Convection-Diffusion equation: -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = 1

Analytical solution: (1/c)*(x-((1-exp(c*x/nu))/(1-exp(c/nu))))

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
from math import pi
%matplotlib qt
#plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.01
c = 2

"Number of points"
N = 1024
Dx = 1/N
x = np.linspace(0,1,N-1)

"Time array"
dt = 0.1; t = np.arange(0,3+dt,dt); nt = t.shape[0]

"Initial Conditions"
u0 = 0
U = np.zeros((N-1,nt))
U[:,0] = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu)))) + u0


flag = 0
for it in range(0,nt-1):
    
    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
       
    "Advection term"
    if(flag): ## Second Order Term
        "Advection term: centered differences"
        Advp = -0.5*c*np.diag(np.ones(N-2),-1)
        Advm = -0.5*c*np.diag(np.ones(N-2),1)
        Adv = (1/Dx)*(Advp-Advm)
        A = Diff + Adv 
    else: ## First Order Term
        Advp = max(c,0)*(np.diag(np.ones(N-1),0) - np.diag(np.ones(N-2),-1) )
        Advm = min(c,0)*(np.diag(np.ones(N-1),0) - np.diag(np.ones(N-2),1) )
        Adv = (1/Dx)*(Advp-Advm)
        A = Diff + Adv
        
    
    "Source term"
    F = np.ones(N-1)
        
    "Temporal Term"
    A = A + (1/dt)*np.diag(np.ones(N-1))
    F = F + U[:,it]/dt
        
    

    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u 
    

ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))

"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(0,1/c)) 
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = np.concatenate(([u0],U[0:N-3,i],[0]))
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,interval=200,frames=range(1,nt),blit=True,repeat=True)



"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);


   