# Functions for the individual state level model with random censoring, common
# variance and negative binomial counts for the number of states for each
# peptide.

import numpy as np
import copy
from scipy import special
from scipy import optimize
from scipy import linalg

#==============================================================================
# Useful constants
#==============================================================================

EPS = np.spacing(1)

#==============================================================================
# Exceptions
#==============================================================================

class Error(Exception):
    '''
    Base class for errors in lib.
    '''
    pass

class BisectionError(Error):
    '''
    Exception class for errors particular to bisection algorithms.
    '''

#==============================================================================
# Densities and probabilities
#==============================================================================

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

def dgamma(x, shape=1., rate=1., log=False):
    '''
    Normalized gamma density function, parameterized by shape and rate.
    '''
    ld = np.log(x)*(shape - 1.) - rate*x
    ld += shape*np.log(rate) - special.gammaln(shape)
    if log:
        return ld
    return np.exp(ld)

def lp_profile_gamma(shape, x, prior_shape=1., prior_rate=0.,
                     prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute profile log-posterior of shape parameter for gamma likelihood.
    Assuming conditionally-conjugate gamma prior on observation distribution's
    rate parameter with given parameters.

    Also using log-normal prior on shape parameter itself with given log-mean
    and precision.

    Returns a float with the profile log-posterior.
    '''
    n = np.size(x)

    # Compute conditional posterior mode of rate parameter
    rate_hat = (shape + (prior_shape-1.)/n)/(np.mean(x) + prior_rate/n)

    # Evaluate log-posterior at conditional mode
    lp = np.sum(dgamma(x, shape=shape, rate=rate_hat, log=True))
    # Add prior for rate parameter
    lp += dgamma(rate_hat, shape=prior_shape, rate=prior_rate, log=True)
    # Add prior for shape parameter
    lp += dlnorm(shape, mu=prior_mean_log,
                 sigmasq=1./np.float64(prior_prec_log), log=True)

    return lp

def dnbinom(x, r, p, log=False):
    '''
    Normalized PMF for negative binomial distribution. Parameterized s.t.
    x >= 0, expectation is p*r/(1-p); variance is p*r/(1-p)**2.
    Syntax mirrors R.
    '''
    ld = (np.log(p)*x + np.log(1-p)*r + special.gammaln(x+r) -
          special.gammaln(x+1) - special.gammaln(r))
    if log:
        return ld
    return np.exp(ld)

def dbeta(x, a, b, log=False):
    '''
    Normalized PDF for beta distribution. Syntax mirrors R.
    '''
    ld = np.log(x)*(a-1.) + np.log(1.-x)*(b-1.) - special.betaln(a,b)
    if log:
        return ld
    return np.exp(ld)

#==============================================================================
# Useful derivatives; primarily used in mode-finding routines
#==============================================================================

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

def score_profile_posterior_gamma(shape, x, log=False,
                                  prior_shape=1., prior_rate=0.,
                                  prior_mean_log=0., prior_prec_log=0.):
    '''
    Profile posterior score for shape parameter of gamma distribution.

    If log, compute score for log(shape) instead.

    Assumes a conjugate gamma prior on the rate parameter and an independent
    log-normal prior on the shape parameter, each with the given parameters.

    Returns a float with the desired score.
    '''
    # Compute conditional posterior mode of rate parameter
    n = np.size(x)
    rate_hat = (shape + (prior_shape-1.)/n)/(np.mean(x) + prior_rate/n)

    # Compute score for untransformed shape parameter
    score = (np.sum(np.log(x)) - n*special.polygamma(0, shape) +
             n*np.log(rate_hat) -
             prior_prec_log*(np.log(shape)-prior_mean_log)/shape - 1./shape)

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Compute derivative of untransformed parameters wrt transformed ones
        deriv   = shape

        # Update information using chain rule
        score   *= deriv

    return score

def info_posterior_gamma(shape, rate, x, log=False,
                         prior_shape=1., prior_rate=0.,
                         prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute posterior information for shape and rate parameters of gamma
    distribution.

    If log, compute information for log(shape) and log(rate) instead.
    This is typically more useful, as the normal approximation holds much better
    on the log scale.

    Assumes a conjugate gamma prior on the rate parameter and an independent
    log-normal prior on the shape parameter, each with the given parameters.

    Returns a 2x2 np.ndarray for which the first {row,column} corresponds to the
    shape parameter and the second corresponds to the rate parameter.
    '''
    # Compute observed information for untransformed parameters
    n = np.size(x)
    info = np.zeros((2,2))

    # shape, shape
    info[0,0] = (n*special.polygamma(1, shape) +
                 1/shape**2*(1+prior_prec_log*(np.log(shape)-prior_mean_log-1)))
    # rate, rate
    info[1,1] = (n*shape+prior_shape-1.)/rate**2
    # shape, rate and rate, shape
    info[0,1] = info[1,0] = -n/rate

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = np.array([-n*np.log(rate) + n*special.polygamma(0, shape) -
                         np.sum(np.log(x)) +
                         prior_prec_log*(np.log(shape)-prior_mean_log)/shape +
                         1./shape,
                         -(n*shape+prior_shape-1.)/rate+np.sum(x)+prior_rate])

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv   = np.array([shape, rate])
        deriv2  = deriv

        # Update information using chain rule
        info    = info * deriv
        info    = (info.T * deriv).T
        np.fill_diagonal(info, info.diagonal() + deriv2*grad)

    return info

def info_profile_posterior_gamma(shape, x, log=False,
                                 prior_shape=1., prior_rate=0.,
                                 prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute profile posterior information for shape parameter of gamma
    distribution.

    If log, compute information for log(shape) instead.
    This is typically more useful, as the normal approximation holds much better
    on the log scale.

    Assumes a conjugate gamma prior on the rate parameter and an independent
    log-normal prior on the shape parameter, each with the given parameters.

    Returns a float with the desired information.
    '''
    n = np.size(x)

    # Compute information for untransformed shape parameter
    info = (n*special.polygamma(1, shape) - n/(shape + (prior_shape-1.)/n) +
            1/shape**2*(1+prior_prec_log*(np.log(shape)-prior_mean_log-1)))

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Compute conditional posterior mode of rate parameter
        rate_hat = (shape + (prior_shape-1.)/n)/(np.mean(x) + prior_rate/n)

        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = (-np.sum(np.log(x)) + n*special.polygamma(0, shape) -
                n*np.log(rate_hat) +
                prior_prec_log*(np.log(shape)-prior_mean_log)/shape + 1./shape)

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv   = shape
        deriv2  = deriv

        # Update information using chain rule
        info    = info * deriv**2
        info    += deriv2*grad

    return info

def score_profile_posterior_nbinom(r, x, log=False, prior_a=1., prior_b=1.,
                                   prior_mean_log=0., prior_prec_log=0.):
    '''
    Profile posterior score for r (convolution) parameter of gamma distribution.

    If log, compute score for log(r) instead.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    Returns a float with the desired score.
    '''
    # Compute conditional posterior mode of p
    n = np.size(x)
    A = np.mean(x) + (prior_a- 1.)/n
    B = r + (prior_b - 1.)/n
    p_hat = A / (A + B)
    
    # Compute score for r
    score = (n*np.log(1.-p_hat) + np.sum(special.polygamma(0, x+r))
             - n*special.polygamma(0,r))
    score += -prior_prec_log*(np.log(r) - prior_mean_log)/r - 1./r

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Compute derivative of untransformed parameters wrt transformed ones
        deriv   = r

        # Update information using chain rule
        score   *= deriv

    return score

def info_posterior_nbinom(r, p, x, transform=False, prior_a=1., prior_b=1.,
                          prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute posterior information for r (convolution) and p parameters of
    negative-binomial distribution.

    If transform, compute information for log(r) and logit(rate) instead.
    This is typically more useful, as the normal approximation holds much better
    on the transformed scale.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    Returns a 2x2 np.ndarray for which the first {row,column} corresponds to r
    and the second corresponds to p.
    '''
    # Compute observed information for untransformed parameters
    n = np.size(x)
    info = np.zeros((2,2))

    # r, r
    info[0,0] = (n*special.polygamma(1, r) - np.sum(special.polygamma(1,x+r))
                 + 1/r**2*(1+prior_prec_log*(np.log(r)-prior_mean_log-1)))
    # p, p
    info[1,1] = (n*r + prior_b - 1.)/(1.-p)**2 + (np.sum(x) + prior_a - 1.)/p**2
    # r, p and p, r
    info[0,1] = info[1,0] = n/(1.-p)

    # Handle log transformation of parameters via simple chain rule
    if transform:
        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = np.array([-n*np.log(1.-p) - np.sum(special.polygamma(0, x+r))
                         + n*special.polygamma(0,r)
                         + prior_prec_log*(np.log(r)-prior_mean_log)/r + 1./r,
                         -(np.sum(x) + prior_a - 1.)/p +
                         (n*r + prior_b - 1.)/(1.-p)])

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv   = np.array([r, p*(1.-p)])
        deriv2  = np.array([r, p*(1.-p)*(2.*p-1.)])

        # Update information using chain rule
        info    = info * deriv
        info    = (info.T * deriv).T
        np.fill_diagonal(info, info.diagonal() + deriv2*grad)

    return info

def info_profile_posterior_nbinom(r, x, log=False,
                                  prior_a=1., prior_b=1.,
                                  prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute profile posterior information for r (convolution) parameter of
    negative-binomial distribution.

    If log, compute information for log(r) instead.
    This is typically more useful, as the normal approximation holds much better
    on the transformed scale.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    Returns a float with the desired information.
    '''
    # Compute information for untransformed r
    n = np.size(x)
    A = np.mean(x) + (prior_a- 1.)/n
    B = r + (prior_b - 1.)/n
    p_hat = A / (A + B)

    info = (n*special.polygamma(1, r) - np.sum(special.polygamma(1,x+r))
            - n * p_hat / B
            + 1/r**2*(1+prior_prec_log*(np.log(r)-prior_mean_log-1)))

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = (-n*np.log(1.-p_hat) - np.sum(special.polygamma(0, x+r))
                + n*special.polygamma(0,r))
        grad += prior_prec_log*(np.log(r) - prior_mean_log)/r + 1./r

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv   = r
        deriv2  = r

        # Update information using chain rule
        info    = info * deriv**2
        info    += deriv2*grad

    return info

#==============================================================================
# RNGs
#==============================================================================

def rmvnorm(n, mu, L):
    '''
    Draw d x n matrix of multivariate normal RVs with mean vector mu (length d)
    and covariance matrix L * L.T.
    '''
    d = L.shape[0]
    z = np.random.randn(d, n)

    y = mu + np.dot(L, z)

    return y

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

#==============================================================================
# Optimization and root-finding routines
#==============================================================================

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


#==============================================================================
# Numerical integration functions
#==============================================================================

def laplaceApprox(f, xhat, info, f_args=tuple(), f_kwargs={}):
    '''
    Computes Laplace approximation to integral of f over real line.
    Takes mode xhat and observed information info as inputs.
    Fully comptatible with Numpy vector arguments so long as f is.
    '''
    integral = np.sqrt(2.*np.pi / info) * f(xhat, *f_args, **f_kwargs)
    return integral

#==============================================================================
# Specialized functions for marginalized missing data draws
#==============================================================================

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
        u               = np.random.uniform(size=np.sum(active))
        active[active]  = u > accept_prob

    # Build output
    out = {'mapping' : mapping,
           'W' : W,
           'intensities' : intensities}
    return out

#==============================================================================
# Generic MH updates
#==============================================================================

def mh_update(prop, prev, log_target_ratio, log_prop_ratio):
    '''
    Execute generic Metropolis-Hastings update.

    Takes proposed parameter prop, previous parameter prev, log ratio of target
    densities log_target_ratio, and log ratio of proposal densities
    log_prop_ratio.

    Returns 2-tuple consisting of updated parameter and boolean indicating
    acceptance.
    '''
    # Compute acceptance probability
    log_accept_prob = log_target_ratio - log_prop_ratio

    # Accept proposal with given probability
    accept = (np.log(np.random.uniform(size=1)) < log_accept_prob)

    if accept:
        return (prop, True)
    else:
        return (prev, False)

#==============================================================================
# Gibbs and MH updates
#==============================================================================

def rgibbs_gamma(mu, tausq, sigmasq, y_bar, n_states):
    '''
    Gibbs update for gamma (peptide-level means) given all other parameters.

    This is a standard conjugate normal draw for a hierarchical model.
    The dimensionality of the draw is determined by the size of y_bar.
    All other inputs must be compatible in size.
    '''
    # Compute the conditional posterior mean and variance of gamma
    post_var    = 1. / ( 1./tausq + n_states/sigmasq )
    post_mean   = post_var * ( mu/tausq + y_bar/sigmasq*n_states)

    # Draw gamma
    gamma = np.random.normal(loc=post_mean, scale=np.sqrt(post_var),
                             size=y_bar.size)
    return gamma

def rgibbs_mu(gamma_bar, tausq, n_peptides, prior_mean=0., prior_prec=0.):
    '''
    Gibbs update for mu (protein-level means) given all other parameters.

    This is a standard conjugate normal draw for a hierarchical model.
    The dimensionality of the draw is determined by the size of gamma_bar.
    All other inputs must be compatible in size.
    '''
    # Compute conditional posterior mean and variance
    post_var    = 1. / (n_peptides/tausq + prior_prec)
    post_mean   = post_var * (gamma_bar*n_peptides/tausq +
                              prior_mean*prior_prec)

    # Draw mu
    mu = np.random.normal(loc=post_mean, scale=np.sqrt(post_var),
                          size=gamma_bar.size)
    return mu

def rgibbs_variances(rss, n, alpha=1., beta=0.):
    '''
    Gibbs update for variances given all other parameters.
    Used for sigmasq (state-level variances) and tausq (peptide-level
    variances).

    This is a standard conjugate inverse-gamma draw for a hierarchical model.

    For sigmasq, the input rss must be the vector of residual sums of squares at
    the state level by protein; that is sum( (intensities - gamma)**2 ) by
    protein. n must be a vector consisting of the total number of states
    observed for each protein.

    For tausq, the input rss must be the vector of residual sums of squares at
    the peptide level by protein; that is sum( (gamma - mu)**2 ) by protein. n
    must be a vector consisting of the number of peptides per protein.

    The dimensionality of the draw is determined by the size of rss.
    All other inputs must be compatible in size.
    '''
    variances = 1. / np.random.gamma(shape=alpha + n/2.,
                                     scale=1./(beta + rss/2.),
                                     size=np.size(rss))
    return variances

def rgibbs_p_rnd_cen(n_rnd_cen, n_states, prior_a=1., prior_b=1.):
    '''
    Gibbs update for p_rnd_cen given all other parameters.

    This is a conjugate beta draw with Bernoulli observations.
    n_rnd_cen must be an integer with the total number of randomly-censored
    states.
    n_states must be an integer with the total number of states (including
    imputed).
    prior_a and prior_b are the parameters of a conjugate beta prior.
    '''
    p_rnd_cen = np.random.beta(a=n_rnd_cen+prior_a,
                               b=n_states-n_rnd_cen+prior_b)
    return p_rnd_cen

def rmh_variance_hyperparams(variances, shape_prev, rate_prev,
                             prior_mean_log=2.65, prior_prec_log=1./0.652**2,
                             prior_shape=1., prior_rate=0.,
                             brent_scale=6., fallback_upper=10000.,
                             profile=False):
    '''
    Metropolis-Hastings steps for variance hyperparameters given all other
    parameters.

    Have normal likelihood, so variance likelihood has the same form as gamma
    distribution. Using a log-normal prior for the shape hyperparameter and
    gamma prior for rate hyperparameter.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log-transformed.

    Can propose from approximation to joint conditional posterior
    (profile=False) or approximately marginalize over the rate parameter (by
    profiling) and propose the rate given shape exactly (profile=True).

    Returns a 3-tuple consisting of the new shape, new rate, and a boolean
    indicating acceptance.
    '''
    # Compute posterior mode for shape and rate using profile log-posterior
    n = np.size(variances)

    # Set upper bound first
    if prior_prec_log > 0:
        upper = np.exp(prior_mean_log + brent_scale/np.sqrt(prior_prec_log))
    else:
        upper = fallback_upper

    # Use Brent method to find root of score function
    shape_hat = optimize.brentq(f=score_profile_posterior_gamma,
                                a=EPS, b=upper,
                                args=(variances, False, prior_shape, prior_rate,
                                      prior_mean_log, prior_prec_log))

    if profile:
        # Propose based on profile posterior for shape and exact conditional
        # posterior for rate.

        # Compute proposal variance
        var_prop = 1./info_profile_posterior_gamma(shape=shape_hat,
                                           x=variances, log=True,
                                           prior_shape=prior_shape,
                                           prior_rate=prior_rate,
                                           prior_mean_log=prior_mean_log,
                                           prior_prec_log=prior_prec_log)

        # Propose shape parameter from log-normal
        shape_prop = shape_hat*np.exp(np.sqrt(var_prop)*np.random.randn(1))

        # Propose rate parameter given shape from exact gamma conditional
        # posterior
        rate_prop = np.random.gamma(shape=n*shape_prop + prior_shape,
                                    scale=1./(np.sum(variances) + prior_rate))

        # Compute log-ratio of proposal densities

        # For proposal, start with log-normal proposal for shape
        log_prop_ratio = (dlnorm(shape_prop, mu=np.log(shape_hat),
                                 sigmasq=var_prop, log=True) -
                          dlnorm(shape_prev, mu=np.log(shape_hat),
                                 sigmasq=var_prop, log=True))
        # Then, add conditional gamma proposal for rate
        log_prop_ratio += (dgamma(rate_prop, shape=n*shape_prop + prior_shape,
                                  rate=np.sum(variances) + prior_rate,
                                  log=True) -
                           dgamma(rate_prev, shape=n*shape_prop + prior_shape,
                                  rate=np.sum(variances) + prior_rate,
                                  log=True))
    else:
        # Propose using a bivariate normal approximate to the joint conditional
        # posterior of (shape, rate)

        # Compute posterior mode of rate
        rate_hat = ((shape_hat + (prior_shape-1.)/n) /
                    (np.mean(variances) + prior_rate/n))

        # Compute posterior information matrix for parameters
        info = info_posterior_gamma(shape=shape_hat, rate=rate_hat,
                                    x=variances, log=True,
                                    prior_shape=prior_shape,
                                    prior_rate=prior_rate,
                                    prior_mean_log=prior_mean_log,
                                    prior_prec_log=prior_prec_log)

        # Cholesky decompose information matrix for bivariate draw and
        # density calculations
        U = linalg.cholesky(info, lower=False)

        # Propose shape and rate parameter jointly
        theta_hat   = np.log(np.array([shape_hat, rate_hat]))
        z_prop      = np.random.randn(2)
        theta_prop  = theta_hat + linalg.solve_triangular(U, z_prop)
        shape_prop, rate_prop = np.exp(theta_prop)

        # Demean and decorrelate previous draws
        theta_prev  = np.log(np.array([shape_prev, rate_prev]))
        z_prev      = np.dot(U, theta_prev - theta_hat)

        # Compute log-ratio of proposal densities

        # These are exponentiated bivariate normals with equivalent covariance
        # matrices, so the resulting Jacobian terms cancel. We are left to
        # contend with the z's and the Jacobian terms resulting from
        # exponentiation.
        log_prop_ratio = np.sum(dnorm(z_prop, log=True) -
                                dnorm(z_prev, log=True))
        log_prop_ratio += -np.sum(theta_prop - theta_prev)

    # Compute log-ratio of target densities.
    # This is equivalent for both proposals.

    # For target, start with the likelihood for the variances
    log_target_ratio = np.sum(dgamma(variances, shape=shape_prop,
                                     rate=rate_prop, log=True) -
                              dgamma(variances, shape=shape_prev,
                                     rate=rate_prev, log=True))
    if prior_prec_log > 0:
        # Add the log-normal prior on the shape parameter
        log_target_ratio += (dlnorm(shape_prop, mu=prior_mean_log,
                                    sigmasq=1./prior_prec_log) -
                             dlnorm(shape_prev, mu=prior_mean_log,
                                    sigmasq=1./prior_prec_log))
    # Add the gamma prior on the rate parameter
    if prior_rate > 0:
        log_target_ratio += (dgamma(rate_prop, shape=prior_shape,
                                    rate=prior_rate, log=True) -
                             dgamma(rate_prev, shape=prior_shape,
                                    rate=prior_rate, log=True))
    else:
        log_target_ratio += np.log(rate_prop/rate_prev)*(shape_prop - 1.)

    # Execute MH update
    return mh_update(prop=(shape_prop, rate_prop), prev=(shape_prev, rate_prev),
                     log_target_ratio=log_target_ratio,
                     log_prop_ratio=log_prop_ratio)

def rmh_nbinom_hyperparams(x, r_prev, p_prev,
                           prior_mean_log=2.65, prior_prec_log=1./0.652**2,
                           prior_a=1., prior_b=1.,
                           brent_scale=6., fallback_upper=10000.,
                           profile=False):
    '''
    Metropolis-Hastings steps for negative-binomial hyperparameters given all
    other parameters.

    Using a log-normal prior for the r (convolution) hyperparameter and a
    conditionall-conjugate beta prior for p.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log-transformed.

    Can propose from approximation to joint conditional posterior
    (profile=False) or approximately marginalize over the p parameter (by
    profiling) and propose the p given shape exactly (profile=True).

    Returns a 3-tuple consisting of the new r, new p, and a boolean indicating
    acceptance.
    '''
    # Compute posterior mode for r and p using profile log-posterior
    n = np.size(x)

    # Set upper bound first
    if prior_prec_log > 0:
        upper = np.exp(prior_mean_log + brent_scale/np.sqrt(prior_prec_log))
    else:
        upper = fallback_upper

    # Use Brent method to find root of score function
    r_hat = optimize.brentq(f=score_profile_posterior_nbinom, a=EPS, b=upper,
                            args=(x, False, prior_a, prior_b,
                                  prior_mean_log, prior_prec_log))

    if profile:
        # Propose based on profile posterior for r and exact conditional
        # posterior for p.

        # Compute proposal variance
        var_prop = 1./info_profile_posterior_nbinom(r=r_hat, x=x, log=True,
                                           prior_a=prior_a, prior_b=prior_b,
                                           prior_mean_log=prior_mean_log,
                                           prior_prec_log=prior_prec_log)

        # Propose r parameter from log-normal
        r_prop = r_hat*np.exp(np.sqrt(var_prop)*np.random.randn(1))

        # Propose p parameter given r from exact beta conditional posterior
        p_prop = np.random.beta(a=np.sum(x) + prior_a - 1.,
                                b=n*r_prop + prior_b - 1.)

        # Compute log-ratio of proposal densities

        # For proposal, start with log-normal proposal for r
        log_prop_ratio = (dlnorm(r_prop, mu=np.log(r_hat),
                                 sigmasq=var_prop, log=True) -
                          dlnorm(r_prev, mu=np.log(r_hat),
                                 sigmasq=var_prop, log=True))
        # Then, add conditional beta proposal for p
        log_prop_ratio += (dbeta(p_prop, a=np.sum(x) + prior_a - 1.,
                                 b=n*r_prop + prior_b - 1.,
                                 log=True) -
                           dbeta(p_prev, a=np.sum(x) + prior_a - 1.,
                                 b=n*r_prev + prior_b - 1.,
                                 log=True))
    else:
        # Propose using a bivariate normal approximate to the joint conditional
        # posterior of (r, p)

        # Compute posterior mode of p
        A = np.mean(x) + (prior_a- 1.)/n
        B = r_hat + (prior_b - 1.)/n
        p_hat = A / (A + B)

        # Compute posterior information matrix for parameters
        info = info_posterior_nbinom(r=r_hat, p=p_hat, x=x, transform=True,
                                     prior_a=prior_a, prior_b=prior_b,
                                     prior_mean_log=prior_mean_log,
                                     prior_prec_log=prior_prec_log)

        # Cholesky decompose information matrix for bivariate draw and
        # density calculations
        U = linalg.cholesky(info, lower=False)

        # Propose shape and rate parameter jointly
        theta_hat   = np.log(np.array([r_hat, p_hat]))
        z_prop      = np.random.randn(2)
        theta_prop  = theta_hat + linalg.solve_triangular(U, z_prop)
        r_prop, p_prop = np.exp(theta_prop)
        p_prop = p_prop / (1. + p_prop)

        # Demean and decorrelate previous draws
        theta_prev  = np.log(np.array([r_prev, p_prev]))
        theta_prev[1] = theta_prev[1] - np.log(1.-p_prev)
        z_prev      = np.dot(U, theta_prev - theta_hat)

        # Compute log-ratio of proposal densities

        # These are transformed bivariate normals with equivalent covariance
        # matrices, so the resulting Jacobian terms cancel. We are left to
        # contend with the z's and the Jacobian terms resulting from the
        # exponential and logit transformations.
        log_prop_ratio = np.sum(dnorm(z_prop, log=True) -
                                dnorm(z_prev, log=True))
        log_prop_ratio += -(np.log(r_prop) - np.log(r_prev))
        log_prop_ratio += -(np.log(p_prop) + np.log(1.-p_prop)
                            -np.log(p_prev) - np.log(1.-p_prev))

    # Compute log-ratio of target densities.
    # This is equivalent for both proposals.

    # For target, start with the likelihood for x
    log_target_ratio = np.sum(dnbinom(x, r=r_prop, p=p_prop, log=True) -
                              dnbinom(x, r=r_prev, p=p_prev, log=True))
    if prior_prec_log > 0:
        # Add the log-normal prior on r
        log_target_ratio += (dlnorm(r_prop, mu=prior_mean_log,
                                    sigmasq=1./prior_prec_log) -
                             dlnorm(r_prev, mu=prior_mean_log,
                                    sigmasq=1./prior_prec_log))
    # Add the beta prior on p
    log_target_ratio += (dbeta(p_prop, a=prior_a, b=prior_b, log=True) -
                         dbeta(p_prev, a=prior_a, b=prior_b, log=True))

    # Execute MH update
    return mh_update(prop=(r_prop, p_prop), prev=(r_prev, p_prev),
                     log_target_ratio=log_target_ratio,
                     log_prop_ratio=log_prop_ratio)
