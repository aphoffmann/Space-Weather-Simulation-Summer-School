#!/usr/bin/env python
"""
Solution of a 1D Convection-Diffusion equation: -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = 1

Analytical solution: (1/c)*(x-((1-exp(c*x/nu))/(1-exp(c/nu))))

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme
    - First-order upwind

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
from math import pi
%matplotlib qt
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.001
c = 2

"Scheme parameters"
beta = 2

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)

"Time parameters"
dt = .05
time = np.arange(0,3+dt,dt)
nt = np.size(time)

"Initialize solution variable"
U = np.zeros((N-1,nt))


for it in range(nt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))

    "Advection term:"
        
    "Sensor"
    U0 = U[:,it]
    uaux = np.concatenate(([0],U0,[0]))
    Du = uaux[1:N+1] - uaux[0:N] + 1e-8
    r = Du[0:N-1]/Du[1:N]
    
    "Limiter"
    if beta>0:
        phi = np.minimum(np.minimum(beta*r,1),np.minimum(r,beta))
        phi = np.maximum(0,phi)
    else:
        phi = 2*r/(r**2 + 1)
        
    phim = phi[0:N-2]
    phip = phi[1:N-1]
        
    
    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    Advp = cp*(np.diag(1-phi) - np.diag(1-phip,-1))
    Advm = cm*(np.diag(1-phi) - np.diag(1-phim,1))
    Alow = Advp-Advm
    "Centered differences"
    Advp = -0.5*c*np.diag(phip,-1)
    Advm = -0.5*c*np.diag(phim,1)
    Ahigh = Advp-Advm
        
    Adv = (1/Dx)*(Ahigh + Alow)
    A = Diff + Adv
    "Source term"
    F = np.ones(N-1)
    
    "Temporal terms"
    A = A + (1/dt)*np.diag(np.ones(N-1))
    F = F + U0/dt


    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = np.concatenate(([0],u,[0]))


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
    
    u = np.concatenate(([0],U[0:N+1,i],[0]))
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True,repeat=True)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*dt/Dx)
print("CFL number CFL=%g\n" % CFL);

