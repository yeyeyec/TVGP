# % ***************************************************************************************
# %
# % JumpGP_LD - The function implements Jump GP with a linear decision boundary function
# %             described in the paper,
# % 
# %   Park, C. (2022) Jump Gaussian Process Model for Estimating Piecewise 
# %   Continuous Regression Functions. Journal of Machine Learning Research.
# %   23. 
# % 
# %
# % Inputs:
# %       x - training inputs
# %       y - training responses
# %       xt - test inputs
# %       mode - inference algorithm. It can be either 
# %                        'CEM' : Classification EM Algorithm
# %                        'VEM' : Variational EM Algorithm
# %                        'SEM' : Stochastic EM Algorithm 
# %       bVerbose (Internal Use for Debugging) 
# %                  0: do not visualize output
# %                  1: visualize output 
# % Outputs:
# %       mu_t - mean prediction at xt
# %       sig2_t - variance prediction at xt
# %       model - fitted JGP model
# %       h     - (internal use only) 
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from cov.covSum import covSum
from cov.covSEard import covSEard
from cov.covNoise import covNoise
from lik.loglikelihood import loglikelihood
from local_linearfit import local_linearfit
from maximize_PD import maximize_PD
from variationalEM import variationalEM
from stochasticEM import stochasticEM
from calculate_gx import calculate_gx

# Main function for JumpGP
def JumpGP_LD(x, y, xt, mode, bVerbose=False, *args):
    """
    JumpGP_LD - The function implements Jump GP with a linear decision boundary function
    
    Parameters:
    -----------
    x : array-like
        Training inputs
    y : array-like
        Training responses
    xt : array-like
        Test inputs
    mode : str
        Inference algorithm. It can be either:
        'CEM' : Classification EM Algorithm
        'VEM' : Variational EM Algorithm
        'SEM' : Stochastic EM Algorithm
    bVerbose : bool, optional
        Whether to visualize output (for debugging)
    args : tuple, optional
        Additional arguments, including logtheta if provided
    
    Returns:
    --------
    mu_t : array-like
        Mean prediction at xt
    sig2_t : array-like
        Variance prediction at xt
    model : dict
        Fitted JGP model
    h : list
        Plot handles (for internal use only)
    """
    cv = [covSum, [covSEard, covNoise]]
    d = x.shape[1]
    px = x
    pxt = xt

    # Initial estimation of the boundary B(x)
    if len(args) > 0 and args[0] is not None:
        logtheta = args[0]
    else:
        logtheta = np.zeros(d + 2)
        logtheta[-1] = -1.15

    w, _ = local_linearfit(x, y, xt[0])  # Use only the first test point for initial fit
    nw = np.linalg.norm(w)
    w = w / nw

    # Fine-tune the intercept term
    w1 = np.asarray(w).ravel()
    b = np.arange(-1 + w1[0], 1 + w1[0], 0.01)
    fd = []
    for bi in b:
        w_d = w.copy()
        w_d[0] = bi
        gx, _ = calculate_gx(px, w_d)
        r = gx >= 0
        a = np.zeros(2)
        if np.sum(~r) > 0:
            a[0] = np.mean(~r) * np.var(y[~r])
        if np.sum(r) > 0:
            a[1] = np.mean(r) * np.var(y[r])
        fd.append(np.sum(a[~np.isnan(a)]))
    
    k = np.argmin(fd)
    w[0] = b[k]
    w = nw * w

    # Select algorithm
    if mode == 'CEM':
        model = maximize_PD(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'VEM':
        model = variationalEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)
    elif mode == 'SEM':
        model = stochasticEM(x, y, xt, px, pxt, w, logtheta, cv, bVerbose)

    
    mu_t = model['mu_t']
    sig2_t = model['sig2_t']

    h = []
    if bVerbose:
        a = np.array([[1, -0.5], [1, 0.5]])
        b = -a @ model['w'][0:2] / model['w'][2]
        h1 = plt.plot(a, b, 'r', linewidth=3)
        gx, _ = calculate_gx(px, model['w'])
        # print("gx", gx.shape)
        gx = gx.ravel()
        h2 = plt.scatter(x[gx >= 0, 0], x[gx >= 0, 1], color='g', marker='s')
        h = [h2, h1[0]]
    
    return mu_t, sig2_t, model, h

# Example usage
if __name__ == "__main__":
    x_train = np.random.rand(100, 2)
    y_train = np.random.rand(100)
    x_test = np.random.rand(20, 2)
    mu_t, sig2_t, model, h = JumpGP_LD(x_train, y_train.reshape(-1, 1), x_test, 'VEM', 1)
    print("Successfully run the example!")
