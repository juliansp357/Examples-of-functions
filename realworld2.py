# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:31:04 2020

@author: marqu
"""

import numpy as np

from scipy.special import jv
import matplotlib.pyplot as plt

x = np.linspace(-20,20,1000)

for i in range(0, 2 + 1):
    J = jv(i,x)
    plt.plot(x,J, label= r'$J' + str(i) + '(x)$')
    

plt.legend()
plt.title('Bessel function', fontweight='bold', fontsize=20)
plt.xlabel('x', fontweight='bold', fontsize=16)
plt.ylabel('$J_v(x)$', fontweight='bold', fontsize=16)
plt.grid()
plt.show()


from scipy.special import spherical_jn
import mpl_toolkits.mplot3d as Axes3D

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

x,y = np.meshgrid(x,y)

#radial distance

r = np.sqrt(x**2 + y**2)

#z variable

z = spherical_jn(0,r)

fig = plt.figure('Spherical Bessel function')
ax = fig.add_subplot(111, projection= '3d')

h = ax.plot_surface(x,y,z, cmap = 'jet', edgecolor='k')
plt.colorbar(h)

ax.set_xlabel('X', fontweight='bold', fontsize=14)
ax.set_ylabel('Y', fontweight='bold', fontsize=14)
ax.set_zlabel('Z', fontweight='bold', fontsize=14)

ax.set_title('Spherical Bessel function', fontweight='bold', fontsize=16)
plt.show()


import sympy as sp
from sympy import fourier_transform, exp,symbols 
from sympy.abc import x, k

a=fourier_transform(exp(-x**2), x, k)
s=symbols('s') 
Ori=(s)*exp(-(x**2)/(s**2)) 
FT=fourier_transform(Ori,x,k)
 
a.subs({k:1}).evalf()


from scipy.special import legendre
 
N = 1000
xval = np.linspace(-1,1,N)
def frank(x,n):
    leg = legendre(n)
    P_n = leg(x)
    return  P_n

func = frank(xval, 1)
func1 = frank(xval,2)
func2 = frank(xval, 3)

plt.plot(xval, func, 'r--', label='n=1')
plt.plot(xval, func1, 'b--', label='n=2')
plt.plot(xval, func2, 'g--', label='n=3')

plt.title("First 3 P_n(x)")
plt.grid
plt.legend
plt.show()



N = 1000
xval = np.linspace(-np.pi,np.pi,N)
def frank(x,n):
    leg = legendre(n)
    P_n = leg(np.cos(x))
    return  P_n

func = frank(xval, 1)
func1 = frank(xval,2)
func2 = frank(xval, 3)

plt.plot(xval, func, 'r--', label='n=1')
plt.plot(xval, func1, 'b--', label='n=2')
plt.plot(xval, func2, 'g--', label='n=3')

plt.title("First 3 P_n(x)")
plt.grid
plt.legend
plt.show()




N = 1000
xval = np.linspace(-np.pi,np.pi,N)
def frank(x,n):
    leg = legendre(n)
    P_n = leg(np.cos(x))
    return  P_n

for i in range(1,16):
    func = frank(xval,i)
    plt.plot(xval,func,label = 'n = ' + str(i))

plt.title("First 3 P_n(x)")
plt.grid
plt.legend
plt.show()


#solving differential equations


from sympy.interactive import printing
printing.init_printing(use_latex=True)
from sympy import *
import sympy as sp

x = sp.symbols('x')
f = sp.Function('f')(x)

diffeq = Eq(f.diff(x,x) - 4*f, -3)
display(diffeq)

dsolve(diffeq,f)


 

