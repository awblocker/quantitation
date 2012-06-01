import numpy as np
from scipy import linalg
import links
import families

# Function for fast WLS
def wls(X, y, w, method='cholesky'):
    """
    Computes WLS regression of y on X with weights w.
    
    Can use Cholesky or QR decomposition.
    
    The 'cholesky' method is extremely fast as it directly solves the normal
    equations; however, it could lead to issues with numerical stability for
    nearly-singular X'X.
    
    The 'qr' method is slower and requires more memory, but it can be more
    numerically stable.

    :rtype: If method=='cholesky', a dictionary with b (coefficients), L
            (Cholesky decomposition of XwtXw), and resid (residuals). If
            method=='qr', R is returned in place of L.
    """
    # Check for validity of method argument        
    if method not in ('cholesky', 'qr'):
        raise ValueError('Method must be \'cholesky\' or \'qr\'.')
    
    # Calculate weighted variables
    sqrt_w = np.sqrt(w)
    Xw = (X.T * sqrt_w).T
    yw = y*sqrt_w
    
    # Obtain estimates with desired method
    if method == 'cholesky':
        # Calculate Xw.T * Xw and Xw.T * yw
        XwtXw = np.dot(Xw.T, Xw)
        Xwtyw = np.dot(Xw.T, yw)
        
        # Solve normal equations using Cholesky decomposition
        # (faster than QR or SVD)
        L = linalg.cholesky(XwtXw, lower=True)
        b = linalg.cho_solve((L, True), Xwtyw)
    else:
        # QR decompose Xw
        Q, R = linalg.qr(Xw, mode='economic')
        
        # Calculate z = Q'y
        z = np.dot(Q.T, yw)
        
        # Solve reduced normal equations
        b = linalg.solve_triangular(R, z, lower=False)
    
    resid = y - np.dot(X, b)
    
    # Return appropriate values
    if method == 'cholesky':
        return {'b':b, 'L':L, 'resid':resid}
    else:
        return {'b':b, 'R':R, 'resid':resid}

def glm(y, X, family, w=1, offset=0, cov=False, tol=1e-8, maxIter=100,
        ls_method='cholesky'):
    '''
    GLM estimation using IRLS
    '''
    # Get dimensions
    n = X.shape[0]
    p = X.shape[1]
    
    # Initalize mu and eta
    mu  = family.mu_init(y)
    eta = family.link(mu)
    
    # Initialize deviance
    dev = family.deviance(y=y, mu=mu, w=w)
    if np.isnan(dev):
        raise ValueError('Deviance is NaN. Boundary case?')
    
    # Initialize for iterations
    dev_last    = dev
    iteration   = 0
    converged   = False
    while iteration < maxIter and not converged:
        # Compute weights for WLS
        weights_ls = w*family.weights(mu)
        
        # Compute surrogate dependent variable for WLS
        z = eta + (y-mu)*family.link.deriv(mu) - offset
        
        # Run WLS with desired method
        fit = wls(X=X, y=z, w=weights_ls, method=ls_method)
        
        # Compute new values of eta and mu
        eta = (z - fit['resid']) + offset
        mu  = family.link.inv(eta)
        
        # Update deviance
        dev = family.deviance(y=y, mu=mu, w=w)
        
        # Check for convergence
        criterion = np.abs(dev - dev_last) / (np.abs(dev_last) + 0.1)
        if (criterion < tol):
            converged = True
        
        dev_last = dev        
        iteration += 1
        
    # Start building return value
    result = {'eta' : eta,
              'mu'  : mu,
              'b'   : fit['b'],
              'deviance' : dev,
              'iteration' : iteration}
    
    # Compute approximate covariance, if requested
    if cov:
        if ls_method=='cholesky':
            V = np.eye(p)
            V = linalg.solve_triangular(fit['L'], V, lower=True)
            V = np.dot(V.T, V)
        else:
            V = np.eye(p)
            V = linalg.solve_triangular(fit['R'], V, lower=False)
            V = np.dot(V, V.T)
        
        result['V'] = V
    
    return result
    
    
    