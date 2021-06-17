# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:54:44 2021

@author: 184277J
"""

import numpy as np
import scipy as sp
from scipy import optimize
from scipy.linalg import svd
import matplotlib.pyplot as plt
import sys
sys.path.append('R:\XRDSF-ROWLEM-SE05479\Rowles\programCode\diffUtil')
import DiffUtil as du 



x_data = np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
y_data = np.array([300,300,1000,350,340,1230,500,360,360,920,365,365,350,1000,375,1050,380,385,385,390,400,395,780,410,420,420,415,435,440,435,455])
e_data = np.sqrt(y_data) #uncertainty in y values


# https://hernandis.me/2020/04/05/three-examples-of-nonlinear-least-squares-fitting-in-python-with-scipy.html

def model(params, x):
    """
    Calculate values of the model given parameter values and x value(s)

    Parameters
    ----------
    params : tuple, list, or numpy.array
        a list of the coefficients of the model: eg values for b and m where y = b + m*x
    x : float or numpy.array
        The values at which you want to evaluate the model.

    Returns
    -------
    float or numpy.array
        the values of the model evaluated using the given params and x value(s)
    """
    a0, a1 = params
    return a0 + a1 * x

def v_f(z):
    """
    This function modifies the values of residuals so that application of least-squares ignores
    the effect of outliers - see https://doi.org/10.1107/S0021889801004332.

    Parameters
    ----------
    z : float
        the value of the residual to be modified.

    Returns
    -------
    float
        modified residual value.
    """  
    return np.where(z <= 0, np.square(z), 6*np.log(z/sp.special.erf(z*0.7071067811865475)) - 1.3547481158683645)


def v_f_2(z):
    """
    This function modifies the values of residuals so that application of least-squares ignores
    the effect of outliers - see https://doi.org/10.1107/S0021889801004332.

    Parameters
    ----------
    z : float
        the value of the residual to be modified.

    Returns
    -------
    float
        modified residual value.
    """
    return np.where(z <= 0, z, np.sqrt(6*np.log(z/sp.special.erf(z*0.7071067811865475)) - 1.3547481158683645))


def objective(params, model_func, data, v_modify_residuals_func = None):   
    """
    This is the function that I want to minimise. It takes the tabulated data for x, y, and e, 
    a model function, and initial parameters and returns a residual array ready for sp.optimize.least_squares

    Also, optionally takes a function to modify residuals if you want to weight things differently
    in different places.

    Parameters
    ----------
    params : tuple, list, or numpy.array
        the coefficients of the model that we want to optimise
    model_func : function
        the function we want to apply to our data. It has the form: f(params, x).
    data : tuple or list
        A tuple or list containing y_data, x_data, and (optionally) error_data: eg (yd, xd, ed).
        yd, xd, and ed are numpy.arrays of the tabulated data we want to fit
    modify_residuals_func : function, optional
        A function to modify the value of residuals. Must be capable of taking the entire residual array at once.
        see - np.vectorize

    Returns
    -------
    r : numpy.array
        an array of residuals ready to be passed to sp.optimize.least_squares.
    """
    if len(data) == 3:
        xd, yd, ed = data
    elif len(data) == 2:
        xd, yd = data
        ed = 1  
        
    r = (yd - model_func(params, xd)) / ed # r is an array of residuals
    
    #modify the residuals based on the weighting function that the paper has
    if v_modify_residuals_func is not None:
        r = v_modify_residuals_func(r)
    
    # return the entire modified residuals array
    return r

def objective_sum(params, model_func, data, modify_residuals_func = None):  
    r = objective(params, model_func, data, modify_residuals_func)
    return np.sum(r)
 

def v_cheb(n, x):   
    return np.cos(n * np.arccos(x))


def bkg_model(params, x):
    """
    Calculate values of the model given parameter values and x value(s)

    Parameters
    ----------
    params : tuple, list, or numpy.array
        a list of the coefficients of the bkg model: eg coefficients for a chebyshev polynomial
    x : float or numpy.array
        The values at which you want to evaluate the model.

    Returns
    -------
    float or numpy.array
        the values of the model evaluated using the given params and x value(s)
    """
    r = np.zeros(len(x))
    for i, p in enumerate(params):
        r += p * v_cheb(i, x)
    return r


def v_normaliseList(nparray):
    """ Given a monotonically increasing x-ordinate array, normalise values in the range -1 <= x <= 1. Is vectorised; can be used with numpy.arrays """
    min_ = nparray[0]
    max_ = nparray[-1]
    r = (2*(nparray - min_)/(max_ - min_)) - 1
    return r


import time


def print_debug(string, printMe = True):
    if printMe:
        print(string)

DEBUG = False

#KH_Al2S3_61_ss001.xye
filenames = ["KH_Al2S3_61_ss001.xye", "KH_Al2S3_61_ss002.xye", "KH_Al2S3_61_ss003.xye", "KH_Al2S3_61_ss004.xye"]

keepGoing = True

for i, file in enumerate(filenames):    
    start_read = time.time()
    dp = du.DiffractionPattern(file)
    x_data = np.array(dp.getAngles())
    y_data = np.array(dp.getIntensities())
    e_data = np.array(dp.getErrors())
    nx_data = v_normaliseList(x_data)
    read_time = time.time() - start_read
    
    print_debug(f"Read took {read_time} seconds.", DEBUG)

    plt.plot(x_data, y_data, label='Data')
    plt.legend(loc='best')
    plt.ylim((-500,15000))
    plt.show()

    
    while i == 0:
        chebyshev_order = int(input("What order Chebyshev do you want to try? "))
        initial_params = np.zeros(chebyshev_order + 1)
    
        print("Fitting bkg...")
        start_model = time.time()
        res = sp.optimize.least_squares(objective, initial_params, method = 'lm', args = [bkg_model, [nx_data, y_data, e_data], v_f_2])
        
        bkg = bkg_model(res.x, nx_data)
        y_new = y_data - bkg
        model_time = time.time() - start_model
        
        plt.plot(x_data, y_new, label='Corrected', color="black", linewidth = 0.75)
        plt.plot(x_data, y_data, label='Data')
        plt.plot(x_data, bkg, label='Background', color="red", linewidth = 0.75)
        plt.legend(loc='best')
        plt.ylim((-500,15000))
        plt.show()
        
        print(f"Modelling took {model_time} seconds.")
        try_again = input("Do you like this fit, and want to continue? (y/n): ")
        if try_again == "y":
            break #out of the while loop
        else:
            pass
    
    if i > 0:
        print("Fitting bkg...")
        start_model = time.time()
        initial_params = res.x #use the previous end point as the next start point
        res = sp.optimize.least_squares(objective,
                                        initial_params,
                                        method = 'lm',
                                        args = [bkg_model,
                                                [nx_data, y_data, e_data],
                                                v_f_2])
        model_time = time.time() - start_model
        print_debug(f"Modelling took {model_time} seconds.", DEBUG)

    bkg = bkg_model(res.x, nx_data)
    y_new = y_data - bkg
    
    dp -= bkg.tolist()
    
    start_write = time.time()
    dp.writeToFile("corr_" + dp.filename)
    write_time = time.time() - start_write
    print_debug(f"Writing took {write_time} seconds.", DEBUG)


#plot the last dataset
plt.plot(x_data, y_data, label='Data')
plt.plot(x_data, bkg, label='Background', color="red", linewidth = 0.75)
plt.plot(x_data, y_new, label='Corrected', color="black", linewidth = 0.75)
plt.legend(loc='best')
plt.ylim((-500,15000))
plt.show()






# This one requires me to write my own chi2 function
# results = sp.optimize.minimize(objective_sum, initial_params, args = (bkg_model, (normaliseList(x_data), y_data, e_data), v_f))


    
    
    # #This is how optimize_curve_fit calcs the covariance matrix, so it should be good enough for me
    # _, s, VT = svd(res.jac, full_matrices=False)
    # threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    # s = s[s > threshold]
    # VT = VT[:s.size]
    # pcov = np.dot(VT.T / s**2, VT) #covariance matrix
    # perr = np.sqrt(np.diag(pcov)) #Stdev of coefficients
    


# x_cheb = np.arange(-0.999, 0.999, 0.001)
# n_cheb = 10
# y_cheb = v_cheb(n_cheb, x_cheb)
# plt.plot(x_cheb, y_cheb, label='Cheb')
# plt.show()
