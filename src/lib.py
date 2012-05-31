# Functions for the individual state level model with random censoring, common
# variance and negative binomial counts for the number of states for each
# peptide.

import numpy as np
import copy

from scipy import linalg
from scipy import stats
from scipy import optimize
from scipy import special

# Exceptions

class Error(Exception):
    '''
    Base class for errors in lib.
    '''
    pass

class BisectionError(Error):
    '''
    Exception class for errors particular to bisection algorithms.
    '''

# Densities and probabilities

def dnorm(x, mu=0, sigmasq=1, log=False):
    '''
    Gaussian density parameterized by mean and variance.
    Syntax mirrors R.
    '''
    ld = -0.5*(np.log(2.*np.pi) + np.log(sigmasq)) - (x-mu)**2 / 2./ sigmasq
    if log:
        return ld
    else:
        return np.exp(ld)

def dlnorm(x, mu=0, sigmasq=1, log=False):
    '''
    Density function for log-normal, parameterized by mean and variance of
    log(x). Syntax mirrors R.
    '''
    ld = dnorm(np.log(x), mu, sigmasq, log=True) - np.log(x)
    if log:
        return ld
    else:
        return np.exp(ld)

def p_censored(x, eta_0, eta_1, log=False):
    '''
    Compute probability of intensity-based censoring.
    '''
    lp = -np.log(1. + np.exp(eta_0 + eta_1*x))
    if log:
        return lp
    else:
        return np.exp(lp)

def p_obs(x, eta_0, eta_1, log=False):
    '''
    Compute 1 - probability of intensity-based censoring.
    '''
    lp = -np.log(1. + np.exp(-eta_0 - eta_1*x))
    if log:
        return lp
    else:
        return np.exp(lp)

def dcensored(x, mu, sigmasq, eta_0, eta_1, log=False):
    '''
    Unnormalized density function for censored log-intensities.
    Integrates to p_censored.
    '''
    ld = dnorm(x, mu, sigmasq, log=True) + p_censored(x, eta_0, eta_1, log=True)
    if log:
        return ld
    else:
        return np.exp(ld)

def dobs(x, mu, sigmasq, eta_0, eta_1, log=False):
    '''
    Unnormalized density function for observed log-intensities.
    Integrates to p_obs.
    '''
    ld = dnorm(x, mu, sigmasq, log=True) + p_obs(x, eta_0, eta_1, log=True)
    if log:
        return ld
    else:
        return np.exp(ld)

def dt(x, mu=0., scale=1., df=1., log=False):
    '''
    Normalized t density function with location parameter mu and scale parameter
    scale.
    '''
    ld = -(df + 1.)/2. * np.log(1. + (x - mu)**2 / scale**2 / df)
    ld -= 0.5*np.log(np.pi * df) + np.log(scale)
    ld += special.gammaln((df + 1.)/2.) - special.gammaln(df/2.)
    if log:
        return ld
    return np.exp(ld)

def densityratio(x, eta_0, eta_1, mu, sigmasq, approx_sd, y_hat, propDf,
                 normalizing_cnst, log=False):
    '''
    Target-proposal ratio for censored intensity rejection sampler.
    '''
    ld = dcensored(x, mu, sigmasq, eta_0, eta_1, log=True)
    ld -= dt(x, mu=y_hat, scale=approx_sd, df=propDf, log=True)
    ld += np.log(normalizing_cnst)
    if log:
        return ld
    return np.exp(ld)
        
# Useful derivatives; primarily used in mode-finding routines
def deriv_logdt(x, mu=0, scale=1, df=1.):
    deriv = -(df + 1.) / (1. + (x - mu)**2 / scale**2 / df)
    deriv *= (x - mu) / scale**2 / df
    return deriv

def deriv_logdcensored(x, mu, sigmasq, eta_0, eta_1):
    deriv = (-1. + 1./(1. + np.exp(eta_0 + eta_1*x))) * eta_1 - (x - mu)/sigmasq
    return deriv

def deriv2_logdcensored(x, mu, sigmasq, eta_0, eta_1):
    deriv2 = (-1./sigmasq - (eta_1**2 * np.exp(eta_0 + eta_1*x) ) /
              (1. + np.exp(eta_0 + eta_1 * x))**2)
    return deriv2

def deriv3_logdcensored(x, mu, sigmasq, eta_0, eta_1):
    deriv3 = ((2. * eta_1**3 *np.exp(2.*eta_0 + 2.*eta_1*x)) /
              (1. + np.exp(eta_0 + eta_1*x))**3
              - (eta_1**3 * np.exp(eta_0 + eta_1*x)) /
              (1. + np.exp(eta_0 + eta_1*x))**2)
    return deriv3

def deriv_logdensityratio(x, eta_0, eta_1, mu, sigmasq, approx_sd, y_hat,
                          propDf):
    '''
    First derivative of the log target-proposal ratio for censored intensity
    rejection sampler.
    '''
    deriv = deriv_logdcensored(x, mu, sigmasq, eta_0, eta_1)
    deriv -= deriv_logdt(x, mu=y_hat, scale=approx_sd, df=propDf)
    return deriv

# RNGs

def rncen(n_obs, p_rnd_cen, p_int_cen, lmbda, r):
    '''
    Draw ncen | y, censoring probabilities, lambda, r.
    Must have n_obs, p_rnd_cen, and p_int_cen as Numpy vectors of same length.
    r and lmbda must be scalars.
    '''
    m = n_obs.size

    # Setup vectors for result and stopping indicators
    active  = np.ones(m, dtype=bool)
    n_cen    = np.zeros(m, dtype=int)

    # Compute probability for geometric component of density
    pgeom   = 1-(1-lmbda)*(p_rnd_cen+(1-p_rnd_cen)*p_int_cen)

    # Compute necessary bound for envelope condition
    bound = np.ones(m)
    if r < 1:
        bound *= (n_obs + r - 1) / n_obs

    # Run rejection sampling iterations
    nIter = 0
    while np.sum(active) > 0:
        # Propose from negative binomial distribution
        # This is almost correct, modulo the 0 vs. 1 minimum non-conjugacy
        prop    = np.random.negative_binomial(n_obs[active] + r, pgeom[active],
                                              size=active.size)

        # Compute acceptance probability; bog standard
        u       = np.random.uniform(size=np.sum(active))
        pAccept = (n_obs[active]+prop) / (n_obs[active]+prop+r-1)*bound[active]

        # Alway accept for n_obs == 0; in that case, our draw is exact
        pAccept[n_obs[active]==0] = 1.0

        # Execute acceptance step and update done indicators
        n_cen[active[u<pAccept]] = prop[u<pAccept]
        active[active] = u>pAccept

        nIter += 1

    # Add one to draws for nObs == 0; needed to meet constraint that all
    # peptides exist in at least one state.
    n_cen = n_cen + (n_obs==0)
    return n_cen

# Optimization and root-finding routines

def vectorizedBisection(f, lower, upper, f_args=tuple(), f_kwargs={},
                        tol=1e-10, maxIter=100, full_output=False):
    '''
    Find vector of roots of vectorized function using bisection.
    f should be a vectorized function that takes arguments x and f_args of
    compatible dimensions.
    In the iterations, f is called as:
        f(x, *f_args, **f_kwargs)
    '''
    # Initialization
    mid = lower/2. + upper/2.
    error = upper/2. - lower/2.

    f_lower = f(lower, *f_args, **f_kwargs)
    f_upper = f(upper, *f_args, **f_kwargs)

    # Check if the starting points are valid
    if np.any(np.sign(f_lower) * np.sign(f_upper) > 0):
        raise BisectionError(('Not all upper and lower bounds produce function'
                              ' values of different signs.'))

    # Iterate until convergence to tolerance
    t = 0
    while t <= maxIter and error.max() > tol:
        # Update function values
        f_mid = f(mid, *f_args, **f_kwargs)

        # Select direction to move
        below = np.sign(f_mid)*np.sign(f_lower) >= 0
        above = np.sign(f_mid)*np.sign(f_lower) <= 0

        # Update bounds and stored function values
        lower[below] = mid[below]
        f_lower[below] = f_mid[below]
        upper[above] = mid[above]
        f_upper[above] = f_mid[above]

        # Update midpoint and error
        mid = lower/2. + upper/2.
        error = upper/2. - lower/2.

        # Update iteration counter
        t += 1

    if full_output:
        return (mid, t)    
    
    return mid

def halley(f, fprime, f2prime, x0, f_args=tuple(), f_kwargs={},
           tol=1e-8, maxIter=200, full_output=False):
    '''
    Implements (vectorized) Halley's method for root finding.
    Requires function and its first two derivatives, all with first argument to
    search over (x) and the same later arguments (provided via f_args and
    f_kwargs). In the iterations, f, fprime, and f2prime are called as:
        f(x, *f_args, **f_kwargs)
    '''
    # Initialization
    t = 0
    x = copy.deepcopy(x0)

    while t < maxIter:
        # Evaluate the function and its derivatives
        fx          = f(x, *f_args, **f_kwargs)
        fprimex     = fprime(x, *f_args, **f_kwargs)
        f2primex    = f2prime(x, *f_args, **f_kwargs)

        # Update value of x
        x   = x - (2.*fx*fprimex) / (2.*fprimex**2 - fx*f2primex)

        # Update iteration counter
        t += 1

        # Convergence based upon absolute function value
        if(max(abs(fx)) < tol):
            break
    
    if full_output:
        return (x, t)
    
    return x
    

# Numerical integration functions

def laplaceApprox(f, xhat, info, f_args=tuple(), f_kwargs={}):
    '''
    Computes Laplace approximation to integral of f over real line.
    Takes mode xhat and observed information info as inputs.
    Fully comptatible with Numpy vector arguments so long as f is.
    '''
    integral = np.sqrt(2.*np.pi / info) * f(xhat, *f_args, **f_kwargs)
    return integral

# Specialized functions

def characterizeCensoredIntensityDist(eta_0, eta_1, mu, sigmasq,
                                      tol=1e-5, maxIter=200, bisectIter=10,
                                      bisectScale=6.):
    '''
    Constructs Gaussian approximation to conditional posterior of censored
    intensity likelihood. Approximates marginal p(censored | params) via Laplace
    approximation.

    Returns dictionary with three entries:
        1) y_hat, the approximate mode of the given conditional distribution
        2) p_int_cen, the approximate probabilities of intensity-based
           censoring
        3) approx_sd, the approximate SDs of the conditional intensity
           distributions
    '''
    # Construct kwargs for calls to densities and their derivatives
    dargs = {'eta_0' : eta_0,
             'eta_1' : eta_1,
             'mu' : mu,
             'sigmasq' : sigmasq}
             
    # 1) Find mode of censored intensity density
    
    # First, start with a bit of bisection to get in basin of attraction for
    # Halley's method
    y_hat = vectorizedBisection(f=deriv_logdcensored, f_kwargs=dargs,
                                lower=mu-bisectScale*np.sqrt(sigmasq),
                                upper=mu+bisectScale*np.sqrt(sigmasq),
                                tol=np.sqrt(tol), maxIter=bisectIter)
    
    # Second, run Halley's method to find the censored intensity distribution's
    # mode to much higher precision.
    y_hat = halley(f=deriv_logdcensored, fprime=deriv2_logdcensored,
                   f2prime=deriv3_logdcensored, f_kwargs=dargs,
                   x0=y_hat, tol=tol, maxIter=maxIter)
    
    # 2) Compute approximate SD of censored intensity distribution
    info        = -deriv2_logdcensored(y_hat, **dargs)
    approx_sd   = np.sqrt(1./info)
    
    # 3) Use Laplace approximation to approximate p(int. censoring); this is the
    # normalizing constant of the given conditional distribution
    p_int_cen = laplaceApprox(f=dcensored, xhat=y_hat, info=info,
                              f_kwargs=dargs)
    
    # Return dictionary containing combined result
    result = {'y_hat' : y_hat,
              'p_int_cen' : p_int_cen,
              'approx_sd' : approx_sd}
    return result
    
def boundDensityRatio(eta_0, eta_1, mu, sigmasq, y_hat, approx_sd, propDf,
                      normalizing_cnst, tol=1e-10, maxIter=100,
                      bisectScale=1.):
    '''
    Bound ratio of t proposal density to actual censored intensity density.
    This is used to construct an efficient, robust rejection sampler to exactly
    draw from the conditional posterior of censored intensities.
    This computation is fully vectorized with respect to mu, sigmasq, y_hat,
    approx_sd, and normalizing_cnst.

    Based on the properties of these two densities, their ratio will have three
    critical points. These consist of a local minimum, flanked by two local
    maxima. 
    
    It returns the smallest constant M such that the t proposal density times M
    is uniformly >= the actual censored intensity density.
    '''
    # Construct kwargs for calls to densities and their derivatives
    dargs = {'eta_0' : eta_0,
             'eta_1' : eta_1,
             'mu' : mu,
             'sigmasq' : sigmasq,
             'approx_sd' : approx_sd,
             'y_hat' : y_hat,
             'propDf' : propDf}
    
    # Initialize vectors for all four of the bounds
    left_lower = np.zeros_like(y_hat)
    left_upper = np.zeros_like(y_hat)
    right_lower = np.zeros_like(y_hat)
    right_upper = np.zeros_like(y_hat)
    
    # Make sure the starting points are the correct sign
    left_lower = y_hat - bisectScale*approx_sd
    left_upper = y_hat - 10*tol
    right_lower = y_hat + 10*tol
    right_upper = y_hat + bisectScale*approx_sd
    
    # Left lower bounds
    invalid = (deriv_logdensityratio(left_lower, **dargs) < 0)
    while np.any(invalid):
        left_lower[invalid] -= approx_sd[invalid]
        invalid = (deriv_logdensityratio(left_lower, **dargs) < 0)
    
    # Left upper bounds
    invalid = (deriv_logdensityratio(left_upper, **dargs) > 0)
    while np.any(invalid):
        left_lower[invalid] -= 10*tol
        invalid = (deriv_logdensityratio(left_upper, **dargs) > 0)
    
    # Right lower bounds
    invalid = (deriv_logdensityratio(right_lower, **dargs) < 0)
    while np.any(invalid):
        right_lower[invalid] += 10*tol
        invalid = (deriv_logdensityratio(right_lower, **dargs) < 0)
    
    # Right upper bounds
    invalid = (deriv_logdensityratio(right_upper, **dargs) > 0)
    while np.any(invalid):
        right_upper[invalid] += approx_sd[invalid]
        invalid = (deriv_logdensityratio(right_upper, **dargs) > 0)


    # Find zeros that are less than y_hat using bisection.
    left_roots = vectorizedBisection(f=deriv_logdensityratio, f_kwargs=dargs,
                                     lower=left_lower, upper=left_upper,
                                     tol=tol, maxIter=maxIter)
    
    # Find zeros that are greater than y_hat using bisection.
    right_roots = vectorizedBisection(f=deriv_logdensityratio, f_kwargs=dargs,
                                     lower=right_lower, upper=right_upper,
                                     tol=tol, maxIter=maxIter)
        
    # Compute bounding factor M
    f_left_roots = densityratio(left_roots, normalizing_cnst=normalizing_cnst,
                                **dargs)
    f_right_roots = densityratio(right_roots, normalizing_cnst=normalizing_cnst,
                                 **dargs)
    
    # Store maximum of each root
    M = np.maximum(f_left_roots, f_right_roots)
        
    # Return results
    return M

def rcensoredintensities(n_cen, mu, sigmasq, y_hat, approx_sd,
                         p_int_cen, p_rnd_cen,
                         eta_0, eta_1, propDf,
                         tol=1e-10, maxIter=100):
    '''
    Draw censored intensities and random censoring indicators given nCen and
    quantities computed from Laplace approximation
    '''
    # Setup data structures for draws
    n_states = np.sum(n_cen)
    # Intensities
    intensities = np.zeros(n_states, dtype=np.float64)
    # And, the vital indexing vector of length sum(n). This can be used for
    # direct referencing to all input vectors to handle the state to peptide
    # mapping
    mapping = np.zeros(n_states, dtype=int)
    
    # Populate index vector
    filled = 0
    for i in xrange(n_cen.size):
        if n_cen[i] > 0:
            # Get slice to insert new data
            pep = slice(filled, filled + n_cen[i])
            
            # Populate index vector
            mapping[pep] = i
            
            # Update filled counter
            filled += n_cen[i]
    
    # Draw the random censoring indicators. Note that W=1 if randomly censored.
    post_p_rnd_cen = p_rnd_cen / (p_rnd_cen + (1.-p_rnd_cen)*p_int_cen)
    W = (np.random.uniform(size=n_states) < post_p_rnd_cen[mapping]).astype(int)
    
    # Drawing censored intensities    
    # First, get the maximum of the target / proposal ratio for each set of
    # unique parameter values (not per state)
    M = boundDensityRatio(eta_0=eta_0, eta_1=eta_1, mu=mu, sigmasq=sigmasq,
                          y_hat=y_hat, approx_sd=approx_sd,
                          normalizing_cnst=1./p_int_cen, propDf=propDf,
                          tol=tol, maxIter=maxIter)
    
    # Next, draw randomly-censored intensities
    intensities[W==1] = np.random.normal(loc=mu[mapping[W==1]],
                                         scale=np.sqrt(sigmasq[mapping[W==1]]),
                                         size=np.sum(W))
    
    # Draw remaining intensity-censored intensities using rejection sampler
    active = (W == 0)
    while( np.sum(active) > 0):
        # Propose from t distribution
        intensities[active] = np.random.standard_t(df=propDf,
                                                   size=np.sum(active))
        intensities[active] *= approx_sd[mapping[active]]
        intensities[active] += y_hat[mapping[active]]
        
        # Compute acceptance probability
        accept_prob = densityratio(intensities[active],
                                   eta_0=eta_0, eta_1=eta_1,
                                   mu=mu[mapping[active]],
                                   sigmasq=sigmasq[mapping[active]],
                                   approx_sd=approx_sd[mapping[active]],
                                   y_hat=y_hat[mapping[active]],
                                   normalizing_cnst=1./
                                   p_int_cen[mapping[active]],
                                   propDf=propDf, log=False)
        accept_prob /= M[mapping[active]]
        
        # Accept draws with given probabilities by marking corresponding active
        # entries False.
        u       = np.random.uniform(size=np.sum(active))
        active[active] = u > accept_prob
    
    # Build output
    out = {'mapping' : mapping,
           'W' : W,
           'intensities' : intensities}
    return out
    
