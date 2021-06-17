# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:46:24 2021

@author: 184277J
"""

from functools import cache
import sys


sys.path.append('R:\XRDSF-ROWLEM-SE05479\Rowles\programCode\diffUtil')
import DiffUtil as du 
import matplotlib.pyplot as plt



class BGRemove:
    
    BRUCKNER = "bruckner"
    CHEBYSHEV = "chebyshev"
    
    def __init__(self, data, smooth = BRUCKNER):        
        self.dp = du.DiffractionPattern(data)
        self.smooth = smooth       
        self.smooth_data = None

    def initialise(self):        
        if self.smooth == BGRemove.BRUCKNER:
            self.initialise_bruckner()

    def initialise_bruckner(self):
        #Bruckner, S. 2000. "Estimation of the Background in Powder Diffraction Patterns through a Robust Smoothing Procedure." Journal of Applied Crystallography 33 (2): 977-979. https://doi.org/10.1107/S0021889800003617.
        self.smooth_data = self.dp.getIntensities()

        int_ave = 0
        int_min = sys.maxsize
        for x in self.smooth_data:
            int_ave += x
            int_min = min(x, int_min)
        int_ave /= len(self.smooth_data)
       
        threshold = int_ave + 2*(int_ave - int_min)
            
        self.smooth_data = [threshold if x > threshold else x for x in self.smooth_data]
        
    def step_bruckner(self, N):
        #Bruckner, S. 2000. "Estimation of the Background in Powder Diffraction Patterns through a Robust Smoothing Procedure." Journal of Applied Crystallography 33 (2): 977-979. https://doi.org/10.1107/S0021889800003617.
        #extend the data by N on either side to help deal with termination effects
        before = [self.smooth_data[0]]*N
        after  = [self.smooth_data[-1]]*N 
        smoothMe = before + self.smooth_data + after
        
        r=[]
        for i in range(N,len(smoothMe)-N):
            tot = 0
            for j in range(i-N, i+N+1):
                if j == 0:
                    continue
                tot+= smoothMe[j] 
            tot /= 2*N
            r.append(tot)
        
        self.smooth_data = [min(r[i], self.smooth_data[i]) for i in range(len(self.smooth_data))]

    def finalise(self):
        self.dp -= self.smooth_data




def main():
    bgr = BGRemove("bgr.xye")
    bgr.initialise()
    
    orig = bgr.dp.getIntensities()
    
    while True:
        plt.plot(bgr.smooth_data, "r")
        plt.plot(orig, "b")
        plt.ylim(3000,10000)
        plt.show()    
        
        value = input("Do you want to continue? (y/n): ")
        
        if value == "y":
            bgr.step_bruckner(100)
        else:
            break
    
    bgr.finalise()
    bgr.dp.writeToFile("output.xye")
    
 
main()    
       
@cache
def cheb(n: int, x: float):
    """
    Return the value of a Chebyshev polynomial of order n, at a value,x.
    
    -1 <= x <= 1

    Parameters
    ----------
    n : int
        the order of the polynomial.
    x : float
        the value at which to find the polynomial.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if x > 1 or x < -1:
        raise ValueError("x out of range (-1,1).")
    
    if n < 0:
        raise ValueError("n is less than zero.")
    
    if not isinstance(n, int):
        raise TypeError("n needs to be an integer.")
    
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return 2*x*cheb(n-1, x) - cheb(n-2, x)

def oneOnX(n, x):
    return (1/x)**n

def normaliseList(lst):
    r = []
    min_ = lst[0]
    max_ = lst[-1]
    
    for i in lst:
        e = (2*(i - min_)/(max_ - min_)) - 1
        r.append(e)
    return r
    
def chebNormList(n, xlst):
    r = []
    
    for x in xlst:
        e = cheb(n, x)
        r.append(e)
    return r

def chebList(n, xlst):
    xlst = normaliseList(xlst)
    return chebNormList(n, xlst)

def oneOnXList(n, xlst):
    r = []    
    for x in xlst:
        e = oneOnX(n, x)
        r.append(e)
    return r



def cheb_bkg(params, x):
    """
    Calculate values of the chebychev polynomial given parameter values and x value(s).
    The size of the param list determines the order of the polynomial, and the value of each param
    scales that chebyshev

    Parameters
    ----------
    params : tuple, list, or numpy.array
        a list of the coefficients for each order of the chebyshev
    x : float or numpy.array
        The values at which you want to evaluate the chebyshev.

    Returns
    -------
    float or numpy.array
        the values of the model evaluated using the given params and x value(s)
    """
    a0, a1 = params
    return a0 + a1 * x


