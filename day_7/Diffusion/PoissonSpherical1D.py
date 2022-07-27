#!/usr/bin/env python
"""
Solution of a 1D Poisson equation in spherical coordinates: -u_rr - (2/r)*u_r = f
    Domain: [r0,r0+1]   IMPORTANT r>0
    BC: u(r0) = u0, u'(r0+1) = 0
    with f = 2*(2*A(rt)/r + B(rt))*erp(rt), being A(rt) = 2*rt^2 + rt - 3, and B(rt) = 2*rt^2 + 5*rt -2
    and rt = r-r0
    
Analytical solution: 2*(rt)*(3-2*(rt))*exp(rt) + u0

Finite differences (FD) discretization: second-order daccuracy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Introduce first-order derivative with second-order accuracy

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
plt.close()

"Number of points"
N = 16
r0 = 13
Dr = 1/N
r = np.linspace(0,1,N+1) + r0
rN = np.concatenate((r, [r[N]+Dr]))
rNt = rN-r0

"Dirichlet boundary condition at r0"
u0 = 1

"System matrir and RHS term"
A = (1/Dr**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
"First-order term"
A = A + (1/Dr)*(np.diag(1/rN[1:N+2],-1) - np.diag(1/rN[0:N+1],1))

"New source term"
Ar = 2*rNt**2 + rNt - 3
Br = 2*rNt**2 + 5*rNt - 2
F = 2*(2*Ar/rN + Br)*np.exp(rNt)

"Boundary condition at r=0"
A[0,:] = np.concatenate(([1], np.zeros(N+1)))
F[0] = u0

"Boundary condition at r=0"
A[N+1,:] = (0.5/Dr)*np.concatenate((np.zeros(N-1),[-1, 0, 1]))
F[N+1] = 0

"Solution of the linear system AU=F"
u = np.linalg.solve(A,F)
u = u[0:N+1]

rt = r-r0
ua = 2*rt*(3-2*rt)*np.exp(rt)+u0

plt.plot(r,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(r,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

