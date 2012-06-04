import warnings

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

def glm(y, X, family, w=1, offset=0, cov=False, info=False,
        tol=1e-8, maxIter=100, ls_method='cholesky'):
    '''
    GLM estimation using IRLS
    
    y must be a vector of length n consisting of responses.
    
    X must be a (n x p) matrix consisting of covariates. It must include a
    column of ones if you want to fit an intercept.
    
    family must be a Family instance.
 
    w is an optional vector of length n consisting of inverse-variance weights.
    
    offset is an optional scalar or vector of length n consisting of offsets for
    the linear predictor term.
    
    cov and info are switches to return the approximate covariance matrix of the 
    coefficients (as V) and the Fisher information matrix for the coefficients
    (as I).
    
    tol is the convergence tolerance for the IRLS iterations. The iterations
    stop when |dev - dev_last| / (|dev| + 0.1) < tol.
    
    maxIter is the maximum number of IRLS iterations to use.
    
    ls_method is the method to use in wls() calls. The only valid values are
    'cholesky' and 'qr'.
    
    This returns a dictionary consisting of:
        - b_hat, the estimated coefficients
        - mu, the fitted means
        - eta, the fitted linear predictor values
        - deviance, the deviance at convergence
        - iterations, the number of iterations used
        - V, the approximate covariance matrix, if cov
        - I, the estimated Fisher information matrix, if info
    '''
    # Get dimensions
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
        z = eta + (y-mu)/family.link.deriv(eta) - offset
        
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
              'b_hat'   : fit['b'],
              'deviance' : dev,
              'iterations' : iteration}
    
    # Compute Fisher information, if requested
    if info:
        if ls_method=='cholesky':
            I = np.dot(fit['L'], fit['L'].T)
        else:
            I = np.dot(fit['R'].T, fit['R'])
        
        result['I'] = I
    
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
    
def mh_update_glm_coef(b_prev, b_hat, y, X, family, w=1, I=None, V=None,
                       propDf=5., prior_log_density=None, prior_args=tuple(),
                       prior_kwargs={}, **kwargs):
    '''
    Execute single Metropolis-Hastings step for GLM coefficients using normal
    approximation to their posterior distribution. Proposes linearly-transformed 
    vector of independent t_propDf random variables.
    
    At least one of I (the Fisher information) and V (the inverse Fisher 
    information) must be provided. If I is provided, V is ignored. It is more
    efficient to provide the information matrix than the covariance matrix.
    
    Assumes a flat prior on the coefficients by default. Aritrary priors can be
    used via the prior_log_density argument, which must be a function taking a
    numpy array (of coefficients) as its first arguments. This function is
    called as:
            prior_log_density(b, *prior_args, **prior_kwargs)
    
    Returns a 2-tuple consisting of the resulting coefficients and a boolean
    indicating acceptance.
    '''
    # Check for valid precision and/or covariance matrix arguments
    if I is None and V is None:
        raise ValueError('I or V must be specified')
    elif I is not None and V is not None:
        warnings.warn('Only Fisher information I will be used')
    
    # Get dimensions
    p = X.shape[1]
    
    # Compute Cholesky decomposition of information matrix
    if I is not None:
        # Easy case; information matrix provided
        L = linalg.cholesky(I, lower=True)
    else:
        # Harder case; have inverse of information matrix
        L = linalg.cholesky(linalg.inv(V), lower=True)
    
    # Propose from linearly-transformed t with appropriate mean and covariance
    z_prop = (np.random.randn(p) / 
              np.sqrt(np.random.gamma(shape=propDf/2., scale=2., size=p) /
                      propDf))
    b_prop = b_hat + linalg.solve_triangular(L.T, z_prop, lower=False)
    
    # Demean and decorrelate previous draw of b
    z_prev = np.dot(L.T, b_prev - b_hat)
    
    # Compute proposed and previous means
    eta_prop = np.dot(X, b_prop)
    eta_prev = np.dot(X, b_prev)
    
    mu_prop = family.link.inv(eta_prop)
    mu_prev = family.link.inv(eta_prev)
    
    # Compute log-ratio of target densities
    log_target_ratio = np.sum(family.loglik(y=y, mu=mu_prop, w=w) -
                              family.loglik(y=y, mu=mu_prev, w=w))
    
    # Add ratio of priors, if prior_log_density is supplied
    if prior_log_density is not None:
        log_target_ratio += (prior_log_density(b_prop, *prior_args,
                                               **prior_kwargs) -
                             prior_log_density(b_prev, *prior_args,
                                               **prior_kwargs))
    
    # Compute log-ratio of proposal densities. This is very easy with the
    # demeaned and decorrelated values z.
    log_prop_ratio = -(propDf+1.)/2.*np.sum(np.log(1. + z_prop**2/propDf)-
                                            np.log(1. + z_prev**2 /propDf))
    
    # Compute acceptance probability
    log_accept_prob = log_target_ratio - log_prop_ratio
    
    # Accept proposal with given probability
    accept = (np.log(np.random.uniform(size=1)) < log_accept_prob)
    
    if accept:
        return (b_prop, True)
    else:
        return (b_prev, False)
        