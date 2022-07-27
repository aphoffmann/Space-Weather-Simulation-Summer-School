#!/usr/bin/env python
"""
Advection of a pollutant subject to a constant velocity

1D Convection-Diffusion equation: u_t -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u'(0) = 0, u(1) = u0
with f = 100*exp(-((x-0.8)/0.01)^2)*((sin(2*pi*t) + abs(sin(2*pi*t)))/2)

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme
    - First-order upwind
    - Limiters to switch from high to low-resolution
    
    
Tasks:
    - See what happens as we change time-step
    - See what happens as we change viscosity
    
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
nu = 0.01
c = -2
u0 = 0

"Scheme parameters"
beta = 0

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xN = np.concatenate(([x[0]-Dx],x))


"Time parameters"
dt = 1/50
time = np.arange(0,3+dt,dt)
nt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+1,nt))

flag = 0; ghost_flag = False
for it in range(nt-1):
    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))

    "Advection term:"
        
    "Sensor"
    U0 = U[:,it]

    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    if(not flag):
        Advp = cp*(3/2*np.diag(np.ones(N+1), 0) - 2*np.diag(np.ones(N), -1) + 1/2*np.diag(np.ones(N-1), -2))
        Advm = cm*(3/2*np.diag(np.ones(N+1), 0) - 2*np.diag(np.ones(N), 1) + 1/2*np.diag(np.ones(N-1), 2))
    else:
        Advp = cp*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1))
        Advm = cm*(np.diag(np.ones(N+1)) - np.diag(np.ones(N),1))
    
    
    
    Adv = (1/Dx)*(Advp - Advm)
    A = Diff + Adv
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.8)/0.01)**2)*sineplus
    
    "Temporal terms"
    A = A + (1/dt)*np.diag(np.ones(N+1))
    F = F + U0/dt

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-2)))
    F[0] = 0

    "Boundary condition at x=1"
    A = np.vstack((A,np.zeros(N)))
    A[N+1,:] = np.concatenate((np.zeros(N-2),[1,-2,1]))
    F[N+1] = 0
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = u[0:N+1]
    



"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True,repeat=False)


if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*dt/Dx)
print("CFL number CFL=%g\n" % CFL);



