# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:43:06 2022

@author: alexp
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def func(x):
    """Generic Function"""
    return(np.cos(x)+x*np.sin(x))

def func_dot(x):
    """Derivative of Generic Function"""
    return(x*np.cos(x))

def func_forward_diff(x0 = -6, x1 = 6, h = 0.25, func = func):
    """
    Parameters
    ----------
    x0 : float, optional
        DESCRIPTION. Initial Value of x. The default is -6.
    x1 : float, optional
        DESCRIPTION. end value of x. The default is 6.
    h : float, optional
        DESCRIPTION. Step size. The default is 0.25.
    func : function, optional
        DESCRIPTION. function to take a derivative of. The default is func.

    Returns
    -------
    (np.array, np. array).

    """
    x_step = x0
    y = np.zeros(0); x = np.zeros(0)
    while(x_step + h <= 6):
        y = np.append(y, (func(x_step+h) - func(x_step)) / h)
        x = np.append(x, x_step)
        x_step += h
        
    return(x,y)
        
def func_backward_diff(x0 = 6, x1 = -6, h = 0.25, func = func):
    """
    Parameters
    ----------
    x0 : float, optional
        DESCRIPTION. Initial Value of x. The default is -6.
    x1 : float, optional
        DESCRIPTION. end value of x. The default is 6.
    h : float, optional
        DESCRIPTION. Step size. The default is 0.25.
    func : function, optional
        DESCRIPTION. function to take a derivative of. The default is func.

    Returns
    -------
    (np.array, np. array).

    """
    x_step = x0
    y = np.zeros(0); x = np.zeros(0)
    while(x_step - h >= -6):
        y = np.append(y, (func(x_step) - func(x_step+h) ) / h)
        x = np.append(x, x_step)
        x_step -= h
        
    return(x,y)     

def func_central_diff(x0 = -6, x1 = 6, h = 0.25, func = func):
    """
    Parameters
    ----------
    x0 : float, optional
        DESCRIPTION. Initial Value of x. The default is -6.
    x1 : float, optional
        DESCRIPTION. end value of x. The default is 6.
    h : float, optional
        DESCRIPTION. Step size. The default is 0.25.
    func : function, optional
        DESCRIPTION. function to take a derivative of. The default is func.

    Returns
    -------
    (np.array, np. array).

    """
    x_step = x0
    y = np.zeros(0); x = np.zeros(0)
    while(x_step + h <= 6):
        y = np.append(y, (func(x_step+h) - func(x_step-h) ) / (2*h))
        x = np.append(x, x_step)
        x_step += h
        
    return(x,y)  

def morning():
    x = np.linspace(-6,6,1000)
    f = func(x); f_d = func_dot(x); (x_steps, y3) = func_forward_diff()
    (x_steps2, y4) = func_backward_diff(); (x_steps3, y5) = func_central_diff(); 
    fig, ax = plt.subplots(1,1)
    #ax.plot(x,f,  label = "Generic Function");
    ax.plot(x,f_d,  label = "Analytic Diff")
    ax.plot(x_steps, y3, label = "Forward Diff")
    ax.plot(x_steps2, np.flip(y4), label = "Backward Diff")
    ax.plot(x_steps3, y5, label = "Central Diff")
    ax.set_xlabel("x")
    ax.legend()
    


def RHS(x,t):
    """Generic Formula"""
    return(-2*x)



def rungeKutta4(func, current_value, current_time, h):
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + h/2*k1,current_time + h/2)
    k3 = RHS(current_value + h/2*k2,current_time + h/2)
    k4 = RHS(current_value + h*k3,current_time + h)
    return((current_value +k1/6+k2/3+k3/3+k4/6)*h)

def afternoon():
    ## Problem Variables
    y0 = 3; t0 = 0; tf = 2; h = 0.2
    t = np.linspace(t0,tf)
    
    ## True Solution
    y_true = odeint(RHS,y0,t)
    
    
    ## Runge-Kutta 1
    y_est = np.array(y0); times = np.zeros(1) # Empty Arrays
    current_time = t0; current_value = y0    # Initial Values
    while(current_time < tf - h):
        ## Solve
        f = current_value + RHS(current_value, current_time)*h
        
        ## Store
        y_est = np.append(y_est,f); 
        times = np.append(times,current_time+h); 
        
        ## Initialize
        current_value = f; current_time += h

    ## Runge-Kutta 2
    y_est2 = np.array(y0); times2 = np.zeros(1) # Empty Arrays
    current_time = t0; current_value = y0       # Initial Values
    while(current_time < tf - h):
        ## Solve
        f = current_value + h*RHS(current_value + h/2* RHS(current_value, current_time),
                                  current_time + h/2)
        
        ## Store
        y_est2 = np.append(y_est2,f); 
        times2 = np.append(times2,current_time+h); 
        
        ## Initialize
        current_value = f; current_time += h
        
    ## Runge-Kutta 4
    y_est3 = np.array(y0); times3 = np.zeros(1) # Empty Arrays
    current_time = t0; current_value = y0       # Initial Values
    while(current_time < tf - h):
        ## Solve
        k1 = RHS(current_value, current_time)
        k2 = RHS(current_value + h/2*k1,current_time + h/2)
        k3 = RHS(current_value + h/2*k2,current_time + h/2)
        k4 = RHS(current_value + h*k3,current_time + h)
        f = current_value +(k1/6+k2/3+k3/3+k4/4)*h
        
        ## Store
        y_est3 = np.append(y_est3,f); 
        times3 = np.append(times3,current_time+h);
        
        ## Initialize
        current_value = f; current_time += h
        
    fig, ax = plt.subplots(1,1)
    ax.plot(t,y_true, label = "Truth")
    ax.plot(times,y_est, label = "Runge-Kutta 1")
    ax.plot(times2,y_est2, label = "Runge-Kutta 2")
    ax.plot(times3,y_est3, label = "Runge-Kutta 4")
    plt.grid()
    ax.set_xlabel("time")
    ax.legend()
    
    
def pendulum(x,t):
    """Dynamics of the pendulum without any constraint"""
    g = 9.81; l = 3;
    x_dot =np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = -g/l*np.sin(x[0])
    return(x_dot)
    
def damped_pendulum(x,t):
    """Dynamics of the pendulum without any constraint"""
    g = 9.81; l = 3;
    x_dot =np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = -g/l*np.sin(x[0]) - 0.3*x[1]
    return(x_dot)

def afternoon2():
    ## Problem Variables
    
    theta0 = np.array([np.pi/3, 0]); t0 = 0; tf = 15; h = 0.2
    t = np.linspace(t0,tf)
    
    ## True Solution
    y_true = odeint(damped_pendulum,theta0,t).T
    
    ## Runge-Kutta 4
    y_est = np.array(theta0); times = np.zeros(1) # Empty Arrays
    current_time = t0; current_value = theta0       # Initial Values
    while(current_time < tf - h):
        ## Solve
        k1 = damped_pendulum(current_value, current_time)
        k2 = damped_pendulum(current_value + h/2*k1,current_time + h/2)
        k3 = damped_pendulum(current_value + h/2*k2,current_time + h/2)
        k4 = damped_pendulum(current_value + h*k3,current_time + h)
        f = current_value + (k1/6+k2/3+k3/3+k4/6)*h
        
        ## Store
        y_est = np.vstack((y_est, f))
        times = np.append(times,current_time+h);
        
        ## Initialize
        current_value = f; current_time += h
    
    ## plot solutions
    fig, ax = plt.subplots(2,1)
    ax[0].plot(t,y_true[0])
    ax[0].plot(times, y_est.T[0])
    ax[1].plot(t,y_true[1])
    ax[1].plot(times, y_est.T[1])
    return

def lorenz63(x, t, sigma = 10, rho = 28, beta = 8/3):
    x_dot = np.zeros(3)
    x_dot[0] = sigma*(x[1]-x[0])
    x_dot[1] = x[0]*(rho-x[2]) - x[1]
    x_dot[2] = x[0]*x[1] - beta*x[2]
    return(x_dot)

def lorenz():
    x0 = np.array([5,5,5]); t0 = 0; tf = 20; t = np.linspace(t0,tf, 1000)
    sigma = 10; rho = 28; beta = 8/3;
    y_true = odeint(lorenz63,x0,t, args = (sigma, rho, beta)).T
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(y_true[0],y_true[1],y_true[2])
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(20):
        x0 = np.array([np.random.uniform(-20,20),np.random.uniform(-30,30),np.random.uniform(0,50)])
        y_true = odeint(lorenz63,x0,t, args = (sigma, rho, beta)).T
        ax.plot3D(y_true[0],y_true[1],y_true[2])