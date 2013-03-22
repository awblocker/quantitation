# Functions for the individual state level model with random censoring, common
# variance and negative binomial counts for the number of states for each
# peptide.
import bz2
import copy
import cPickle
import gzip
import itertools

import h5py

import numpy as np
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
    ld = -0.5 * (np.log(2. * np.pi) + np.log(sigmasq)) - (x - mu) ** 2 / \
        2. / sigmasq
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
    lp = -np.log(1. + np.exp(eta_0 + eta_1 * x))
    if log:
        return lp
    else:
        return np.exp(lp)


def p_obs(x, eta_0, eta_1, log=False):
    '''
    Compute 1 - probability of intensity-based censoring.
    '''
    lp = -np.log(1. + np.exp(-eta_0 - eta_1 * x))
    if log:
        return lp
    else:
        return np.exp(lp)


def dcensored(x, mu, sigmasq, eta_0, eta_1, log=False):
    '''
    Unnormalized density function for censored log-intensities.
    Integrates to p_censored.
    '''
    ld = dnorm(
        x, mu, sigmasq, log=True) + p_censored(x, eta_0, eta_1, log=True)
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
    ld = -(df + 1.) / 2. * np.log(1. + (x - mu) ** 2 / scale ** 2 / df)
    ld -= 0.5 * np.log(np.pi * df) + np.log(scale)
    ld += special.gammaln((df + 1.) / 2.) - special.gammaln(df / 2.)
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
    ld = np.log(x) * (shape - 1.) - rate * x
    ld += shape * np.log(rate) - special.gammaln(shape)
    if log:
        return ld
    return np.exp(ld)


def lp_profile_gamma(shape, x, log=False, prior_shape=1., prior_rate=0.,
                     prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute profile log-posterior of shape parameter for gamma likelihood.
    Assuming conditionally-conjugate gamma prior on observation distribution's
    rate parameter with given parameters.

    Also using log-normal prior on shape parameter itself with given log-mean
    and precision.

    If log, compute log-posterior for log(shape) and log(rate)

    Returns a float with the profile log-posterior.
    '''
    n = np.size(x)

    # Compute conditional posterior mode of rate parameter
    rate_hat = (
        shape + (prior_shape - 1. + log) / n) / (np.mean(x) + prior_rate / n)

    # Evaluate log-posterior at conditional mode
    lp = np.sum(dgamma(x, shape=shape, rate=rate_hat, log=True))
    # Add prior for rate parameter
    lp += dgamma(rate_hat, shape=prior_shape, rate=prior_rate, log=True)
    # Add prior for shape parameter
    lp += dlnorm(shape, mu=prior_mean_log,
                 sigmasq=1. / np.float64(prior_prec_log), log=True)

    if log:
        # Add Jacobians
        lp += 1. / shape + 1. / rate_hat

    return lp


def dnbinom(x, r, p, log=False):
    '''
    Normalized PMF for negative binomial distribution. Parameterized s.t.
    x >= 0, expectation is p*r/(1-p); variance is p*r/(1-p)**2.
    Syntax mirrors R.
    '''
    ld = (np.log(p) * x + np.log(1 - p) * r + special.gammaln(x + r) -
          special.gammaln(x + 1) - special.gammaln(r))
    if log:
        return ld
    return np.exp(ld)


def dbeta(x, a, b, log=False):
    '''
    Normalized PDF for beta distribution. Syntax mirrors R.
    '''
    ld = np.log(
        x) * (a - 1.) + np.log(1. - x) * (b - 1.) - special.betaln(a, b)
    if log:
        return ld
    return np.exp(ld)

#==============================================================================
# Useful derivatives; primarily used in mode-finding routines
#==============================================================================


def deriv_logdt(x, mu=0, scale=1, df=1.):
    deriv = -(df + 1.) / (1. + (x - mu) ** 2 / scale ** 2 / df)
    deriv *= (x - mu) / scale ** 2 / df
    return deriv


def deriv_logdcensored(x, mu, sigmasq, eta_0, eta_1):
    deriv = (
        -1. + 1. / (1. + np.exp(eta_0 + eta_1 * x))) * eta_1 - (x - mu) / sigmasq
    return deriv


def deriv2_logdcensored(x, mu, sigmasq, eta_0, eta_1):
    deriv2 = (-1. / sigmasq - (eta_1 ** 2 * np.exp(eta_0 + eta_1 * x)) /
              (1. + np.exp(eta_0 + eta_1 * x)) ** 2)
    return deriv2


def deriv3_logdcensored(x, mu, sigmasq, eta_0, eta_1):
    deriv3 = ((2. * eta_1 ** 3 * np.exp(2. * eta_0 + 2. * eta_1 * x)) /
              (1. + np.exp(eta_0 + eta_1 * x)) ** 3
              - (eta_1 ** 3 * np.exp(eta_0 + eta_1 * x)) /
              (1. + np.exp(eta_0 + eta_1 * x)) ** 2)
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


def score_profile_posterior_gamma(shape, x, T=None, log=False,
                                  prior_shape=1., prior_rate=0.,
                                  prior_mean_log=0., prior_prec_log=0.,
                                  prior_adj=1.):
    '''
    Profile posterior score for shape parameter of gamma distribution.

    If log, compute score for log(shape) instead.

    Assumes a conjugate gamma prior on the rate parameter and an independent
    log-normal prior on the shape parameter, each with the given parameters.

    Returns a float with the desired score.
    '''
    # Extract sufficient statistics if needed
    if T is None:
        # Sufficient statistics are (sum x, sum log x, and n)
        T = np.array([np.sum(x), np.sum(np.log(x)), np.size(x)])

    n = T[2]

    # Compute conditional posterior mode of rate parameter
    rate_hat = ((shape + (prior_shape - 1. + log) / n / prior_adj) /
                (T[0]/n + prior_rate / n / prior_adj))

    # Compute score for untransformed shape parameter
    score = (T[1] - n * special.polygamma(0, shape) + n * np.log(rate_hat) -
             prior_prec_log * (np.log(shape) - prior_mean_log) / shape /
             prior_adj - 1. / shape / prior_adj)

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Add Jacobian term
        score += 1. / shape / prior_adj

        # Compute derivative of untransformed parameters wrt transformed ones
        deriv = shape

        # Update information using chain rule
        score *= deriv

    return score


def info_posterior_gamma(shape, rate, x, T=None, log=False,
                         prior_shape=1., prior_rate=0.,
                         prior_mean_log=0., prior_prec_log=0.,
                         prior_adj=1.):
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
    # Extract sufficient statistics if needed
    if T is None:
        # Sufficient statistics are (sum x, sum log x, and n)
        T = np.array([np.sum(x), np.sum(np.log(x)), np.size(x)])

    n = T[2]
    # Compute observed information for untransformed parameters
    info = np.zeros((2, 2))

    # shape, shape
    info[0, 0] = (n * special.polygamma(1, shape) -
                  1 / shape ** 2 * (1 + prior_prec_log *
                 (np.log(shape) - prior_mean_log - 1.)) /
        prior_adj)
    # rate, rate
    info[1, 1] = (n * shape + (prior_shape - 1.) / prior_adj) / rate ** 2
    # shape, rate and rate, shape
    info[0, 1] = info[1, 0] = -n / rate

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Add Jacobian terms
        info[0, 0] += 1. / shape ** 2 / prior_adj
        info[1, 1] += 1. / rate ** 2 / prior_adj

        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = np.array([-n * np.log(rate) + n * special.polygamma(0, shape) -
                         T[1] + prior_prec_log / prior_adj *
                         (np.log(shape) - prior_mean_log) / shape + 1. / shape -
                         log * 1. / shape,
                         -(n * shape + (prior_shape - 1.) / prior_adj) / rate +
                         T[0] + prior_rate / prior_adj - log * 1. / rate])

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv = np.array([shape, rate])
        deriv2 = deriv

        # Update information using chain rule
        info = info * deriv
        info = (info.T * deriv).T
        np.fill_diagonal(info, info.diagonal() + deriv2 * grad)

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
    info = (n * special.polygamma(1, shape) - n / (shape + (prior_shape - 1.) / n) -
            1 / shape ** 2 * (1 + prior_prec_log * (np.log(shape) - prior_mean_log - 1)))

    # Handle log transformation of parameters via simple chain rule
    if log:
        # Compute conditional posterior mode of rate parameter
        rate_hat = ((shape + (prior_shape - 1. + log) / n) /
                    (np.mean(x) + prior_rate / n))

        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = (-np.sum(np.log(x)) + n * special.polygamma(0, shape) -
                n * np.log(rate_hat) +
                prior_prec_log * (np.log(shape) - prior_mean_log) / shape + 1. / shape -
                log * 1. / shape)

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv = shape
        deriv2 = deriv

        # Update information using chain rule
        info = info * deriv ** 2
        info += deriv2 * grad

    return info


def score_profile_posterior_nbinom(r, x, transform=False,
                                   prior_a=1., prior_b=1.,
                                   prior_mean_log=0., prior_prec_log=0.,
                                   prior_adj=1.):
    '''
    Profile posterior score for r (convolution) parameter of negative-binomial
    distribution.

    If transform, compute profile score for log(r) and logit(p) instead.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    The entire log-prior is divided by prior_adj. This is useful for
    constructing distributed approximations.

    Returns a float with the desired score.
    '''
    # Compute conditional posterior mode of p
    n = np.size(x)
    A = np.mean(x) + (prior_a - 1. + transform) / n / prior_adj
    B = r + (prior_b - 1. + transform) / n / prior_adj
    p_hat = A / (A + B)

    # Compute score for r
    # Likelihood
    score = (n * np.log(1. - p_hat) + np.sum(special.polygamma(0, x + r))
             - n * special.polygamma(0, r))
    # Prior
    score += (
        -prior_prec_log * (np.log(r) - prior_mean_log) / r - 1. / r) / prior_adj

    # Handle log transformation of parameters via simple chain rule
    if transform:
        # Add Jacobian term
        score += 1. / r / prior_adj

        # Compute derivative of untransformed parameters wrt transformed ones
        deriv = r

        # Update information using chain rule
        score *= deriv

    return score


def score_posterior_nbinom_vec(theta, x, prior_a=1., prior_b=1.,
                               prior_mean_log=0., prior_prec_log=0.,
                               prior_adj=1.):
    '''
    Posterior score for theta = (log r, logit p) parameter of
    negative-binomial distribution.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    The entire log-prior is divided by prior_adj. This is useful for
    constructing distributed approximations.

    Returns a 2 x m ndarray with the requested score.
    '''
    n = np.size(x)
    if len(np.shape(theta)) < 2:
        theta = theta[:, np.newaxis]

    r = np.exp(theta[0])
    p = 1. / (1. + np.exp(-theta[1]))

    # Compute scores
    xpr = np.ones((theta.shape[1], n)) * x
    xpr = (xpr.T + r).T

    score = np.zeros_like(theta)
    score[0] = (np.sum(special.polygamma(0, xpr), 1) - \
            n * special.polygamma(0, r) + n * np.log(1. - p)) * r + \
            -prior_prec_log * (np.log(r) - prior_mean_log) / prior_adj
    score[1] = (-n * r / (1. - p) + np.sum(x) / p + 
                (prior_a * np.log(p) + prior_b * np.log(1. - p)) /
                prior_adj) * p * (1. - p)

    return score


def info_posterior_nbinom(r, p, x, transform=False, prior_a=1., prior_b=1.,
                          prior_mean_log=0., prior_prec_log=0., prior_adj=1.):
    '''
    Compute posterior information for r (convolution) and p parameters of
    negative-binomial distribution.

    If transform, compute information for log(r) and logit(p) instead.
    This is typically more useful, as the normal approximation holds much better
    on the transformed scale.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    The entire log-prior is divided by prior_adj. This is useful for
    constructing distributed approximations.

    Returns a 2x2 np.ndarray for which the first {row,column} corresponds to r
    and the second corresponds to p.
    '''
    # Compute observed information for untransformed parameters
    n = np.size(x)
    info = np.zeros((2, 2))

    # r, r
    info[0, 0] = (n * special.polygamma(1, r) - np.sum(special.polygamma(1, x + r))
                  - 1 / r ** 2 * (1 + prior_prec_log * (np.log(r) - prior_mean_log - 1)) /
                  prior_adj)
    # p, p
    info[1, 1] = ((n * r + (prior_b - 1.) / prior_adj) / (1. - p) ** 2 +
                 (np.sum(x) + (prior_a - 1.) / prior_adj) / p ** 2)
    # r, p and p, r
    info[0, 1] = info[1, 0] = n / (1. - p)

    # Handle log transformation of parameters via simple chain rule
    if transform:
        # Add Jacobian terms
        info[0, 0] += 1. / r ** 2 / prior_adj
        info[1, 1] += (1. - 2. * p) / p ** 2 / (1. - p) ** 2 / prior_adj

        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = np.array([-n * np.log(1. - p) - np.sum(special.polygamma(0, x + r))
                         + n * special.polygamma(0, r)
                         + (prior_prec_log * (np.log(r) - prior_mean_log) / r + 1. / r -
                         transform * 1. / r) / prior_adj,
                         -(np.sum(x) + (prior_a - 1.) / prior_adj) / p +
                         (n * r + (prior_b - 1.) / prior_adj) / (1. - p) -
                         transform * 1. / p / (1. - p)])

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv = np.array([r, p * (1. - p)])
        deriv2 = np.array([r, p * (1. - p) * (2. * p - 1.)])

        # Update information using chain rule
        info = info * deriv
        info = (info.T * deriv).T
        np.fill_diagonal(info, info.diagonal() + deriv2 * grad)

    return info


def info_profile_posterior_nbinom(r, x, transform=False,
                                  prior_a=1., prior_b=1.,
                                  prior_mean_log=0., prior_prec_log=0.):
    '''
    Compute profile posterior information for r (convolution) parameter of
    negative-binomial distribution.

    If transform, compute profile information for log(r) and logit(p) instead.
    This is typically more useful, as the normal approximation holds much better
    on the transformed scale.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    Returns a float with the desired information.
    '''
    # Compute information for untransformed r
    n = np.size(x)
    A = np.mean(x) + (prior_a - 1. + transform) / n
    B = r + (prior_b - 1. + transform) / n
    p_hat = A / (A + B)

    info = (n * special.polygamma(1, r) - np.sum(special.polygamma(1, x + r))
            - n * p_hat / B
            - 1 / r ** 2 * (1 + prior_prec_log * (np.log(r) - prior_mean_log - 1)))

    # Handle log transformation of parameters via simple chain rule
    if transform:
        # Add Jacobian terms
        info += 1. / r ** 2

        # Compute gradient for log-likelihood wrt untransformed parameters
        grad = (-n * np.log(1. - p_hat) - np.sum(special.polygamma(0, x + r))
                + n * special.polygamma(0, r))
        grad += prior_prec_log * (np.log(r) - prior_mean_log) / r + 2. / r

        # Compute derivatives of untransformed parameters wrt transformed ones
        deriv = r
        deriv2 = r

        # Update information using chain rule
        info = info * deriv ** 2
        info += deriv2 * grad

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
    m = np.size(n_obs)

    # Setup vectors for result and stopping indicators
    active = np.ones(m, dtype=bool)
    n_cen = np.zeros(m, dtype=int)

    # Compute probability for geometric component of density
    pgeom = 1. - (1. - lmbda) * (p_rnd_cen + (1. - p_rnd_cen) * p_int_cen)

    # Compute necessary bound for envelope condition
    bound = np.ones(m)
    if r < 1:
        bound[n_obs > 0] *= (n_obs[n_obs > 0] + r - 1) / n_obs[n_obs > 0]

    # Run rejection sampling iterations
    nIter = 0
    while np.sum(active) > 0:
        # Propose from negative binomial distribution
        # This is almost correct, modulo the 0 vs. 1 minimum non-conjugacy
        prop = np.random.negative_binomial(n_obs[active] + r, pgeom[active],
                                           size=np.sum(active))

        # Compute acceptance probability; bog standard
        u = np.random.uniform(size=np.sum(active))
        pAccept = (
            n_obs[active] + prop) / (n_obs[active] + prop + r - 1) * bound[active]

        # Alway accept for n_obs == 0; in that case, our draw is exact
        pAccept[n_obs[active] == 0] = 1.0

        # Execute acceptance step and update done indicators
        n_cen[active[u < pAccept]] = prop[u < pAccept]
        active[active] = u > pAccept

        nIter += 1

    # Add one to draws for nObs == 0; needed to meet constraint that all
    # peptides exist in at least one state.
    n_cen = n_cen + (n_obs == 0)
    return n_cen

#==============================================================================
# Optimization and root-finding routines
#==============================================================================


def vectorized_bisection(f, lower, upper, f_args=tuple(), f_kwargs={},
                         tol=1e-10, maxIter=100, full_output=False):
    '''
    Find vector of roots of vectorized function using bisection.
    f should be a vectorized function that takes arguments x and f_args of
    compatible dimensions.
    In the iterations, f is called as:
        f(x, *f_args, **f_kwargs)
    '''
    # Initialization
    mid = lower / 2. + upper / 2.
    error = upper / 2. - lower / 2.

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
        below = np.sign(f_mid) * np.sign(f_lower) >= 0
        above = np.sign(f_mid) * np.sign(f_lower) <= 0

        # Update bounds and stored function values
        lower[below] = mid[below]
        f_lower[below] = f_mid[below]
        upper[above] = mid[above]
        f_upper[above] = f_mid[above]

        # Update midpoint and error
        mid = lower / 2. + upper / 2.
        error = upper / 2. - lower / 2.

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
        fx = f(x, *f_args, **f_kwargs)
        fprimex = fprime(x, *f_args, **f_kwargs)
        f2primex = f2prime(x, *f_args, **f_kwargs)

        # Update value of x
        x = x - (2. * fx * fprimex) / (2. * fprimex ** 2 - fx * f2primex)

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

def laplace_approx(f, xhat, info, f_args=tuple(), f_kwargs={}):
    '''
    Computes Laplace approximation to integral of f over real line.
    Takes mode xhat and observed information info as inputs.
    Fully compatible with Numpy vector arguments so long as f is.
    '''
    integral = np.sqrt(2. * np.pi / info) * f(xhat, *f_args, **f_kwargs)
    return integral

#==============================================================================
# Functions for commonly-used MAP estimates
#==============================================================================


def map_estimator_gamma(x, T=None, log=False, prior_shape=1., prior_rate=0.,
                        prior_mean_log=0., prior_prec_log=0., prior_adj=1.,
                        brent_scale=6., fallback_upper=10000.):
    '''
    Maximum a posteriori estimator for shape and rate parameters of gamma
    distribution. If log, compute posterior mode for log(shape) and
    log(rate) instead.

    Assumes a conjugate gamma prior on the rate parameter and an independent
    log-normal prior on the shape parameter, each with the given parameters.

    Returns a 2-tuple with the MAP estimators for shape and rate.
    '''
    # Extract sufficient statistics if needed
    if T is None:
        # Sufficient statistics are (sum 1/variances, sum log 1/variances, and
        # n)
        T = np.array([np.sum(x), np.sum(np.log(x)), np.size(x)])

    # Set upper bound first
    if prior_prec_log > 0:
        upper = np.exp(prior_mean_log + brent_scale / np.sqrt(prior_prec_log))
    else:
        upper = fallback_upper

    # Verify that score is negative at upper bound
    args = (None, T, log, prior_shape, prior_rate, prior_mean_log,
            prior_prec_log, prior_adj)
    while score_profile_posterior_gamma(upper, *args) > 0:
        upper *= 2.

    # Use Brent method to find root of score function
    shape_hat = optimize.brentq(f=score_profile_posterior_gamma,
                                a=np.sqrt(EPS), b=upper,
                                args=args)

    # Compute posterior mode of rate
    rate_hat = ((shape_hat + (prior_shape - 1. + log) / prior_adj / T[2]) /
                (T[0] / T[2] + prior_rate / prior_adj / T[2]))

    return (shape_hat, rate_hat)


def map_estimator_nbinom(x, prior_a=1., prior_b=1., transform=False,
                         prior_mean_log=0., prior_prec_log=0., prior_adj=1.,
                         brent_scale=6., fallback_upper=10000.):
    '''
    Maximum a posteriori estimator for r (convolution) parameter and p parameter
    of negative binomial distribution. If transform, compute posterior mode for
    log(r) and logit(p) instead.

    Assumes a conditionally conjugate beta prior on p and an independent
    log-normal prior on r, each with the given parameters.

    Returns a 2-tuple with the MAP estimators for r and p.
    '''
    # Compute posterior mode for r and p using profile log-posterior
    n = np.size(x)

    # Set upper bound first
    if prior_prec_log > 0:
        upper = np.exp(prior_mean_log + brent_scale / np.sqrt(prior_prec_log))
    else:
        upper = fallback_upper

    # Verify that score is negative at upper bound
    args = (x, transform, prior_a, prior_b, prior_mean_log, prior_prec_log,
            prior_adj)
    while score_profile_posterior_nbinom(upper, *args) > 0:
        upper *= 2.

    # Use Brent method to find root of score function
    r_hat = optimize.brentq(f=score_profile_posterior_nbinom,
                            a=np.sqrt(EPS), b=upper,
                            args=args)

    # Compute posterior mode of p
    A = np.mean(x) + (prior_a - 1. + transform) / n / prior_adj
    B = r_hat + (prior_b - 1. + transform) / n / prior_adj
    p_hat = A / (A + B)

    return (r_hat, p_hat)


#==============================================================================
# Specialized functions for marginalized missing data draws
#==============================================================================

def characterize_censored_intensity_dist(eta_0, eta_1, mu, sigmasq,
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
    dargs = {'eta_0': eta_0,
             'eta_1': eta_1,
             'mu': mu,
             'sigmasq': sigmasq}

    # 1) Find mode of censored intensity density

    # First, start with a bit of bisection to get in basin of attraction for
    # Halley's method

    lower = mu - bisectScale * np.sqrt(sigmasq)
    upper = mu + bisectScale * np.sqrt(sigmasq)

    # Make sure the starting points are of opposite signs
    invalid = (np.sign(deriv_logdcensored(lower, **dargs)) *
               np.sign(deriv_logdcensored(upper, **dargs)) > 0)
    while np.any(invalid):
        lower -= bisectScale * np.sqrt(sigmasq)
        upper += bisectScale * np.sqrt(sigmasq)
        invalid = (np.sign(deriv_logdcensored(lower, **dargs)) *
                   np.sign(deriv_logdcensored(upper, **dargs)) > 0)

    # Run bisection
    y_hat = vectorized_bisection(f=deriv_logdcensored, f_kwargs=dargs,
                                 lower=lower, upper=upper,
                                 tol=np.sqrt(tol), maxIter=bisectIter)

    # Second, run Halley's method to find the censored intensity distribution's
    # mode to much higher precision.
    y_hat = halley(f=deriv_logdcensored, fprime=deriv2_logdcensored,
                   f2prime=deriv3_logdcensored, f_kwargs=dargs,
                   x0=y_hat, tol=tol, maxIter=maxIter)

    # 2) Compute approximate SD of censored intensity distribution
    info = -deriv2_logdcensored(y_hat, **dargs)
    approx_sd = np.sqrt(1. / info)

    # 3) Use Laplace approximation to approximate p(int. censoring); this is the
    # normalizing constant of the given conditional distribution
    p_int_cen = laplace_approx(f=dcensored, xhat=y_hat, info=info,
                               f_kwargs=dargs)

    # Return dictionary containing combined result
    result = {'y_hat': y_hat,
              'p_int_cen': p_int_cen,
              'approx_sd': approx_sd}
    return result


def bound_density_ratio(eta_0, eta_1, mu, sigmasq, y_hat, approx_sd, propDf,
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
    dargs = {'eta_0': eta_0,
             'eta_1': eta_1,
             'mu': mu,
             'sigmasq': sigmasq,
             'approx_sd': approx_sd,
             'y_hat': y_hat,
             'propDf': propDf}

    # Initialize vectors for all four of the bounds
    left_lower = np.zeros_like(y_hat)
    left_upper = np.zeros_like(y_hat)
    right_lower = np.zeros_like(y_hat)
    right_upper = np.zeros_like(y_hat)

    # Make sure the starting points are the correct sign
    left_lower = y_hat - bisectScale * approx_sd
    left_upper = y_hat - 10 * tol
    right_lower = y_hat + 10 * tol
    right_upper = y_hat + bisectScale * approx_sd

    # Left lower bounds
    invalid = (deriv_logdensityratio(left_lower, **dargs) < 0)
    while np.any(invalid):
        left_lower[invalid] -= approx_sd[invalid]
        invalid = (deriv_logdensityratio(left_lower, **dargs) < 0)

    # Left upper bounds
    invalid = (deriv_logdensityratio(left_upper, **dargs) > 0)
    while np.any(invalid):
        left_lower[invalid] -= 10 * tol
        invalid = (deriv_logdensityratio(left_upper, **dargs) > 0)

    # Right lower bounds
    invalid = (deriv_logdensityratio(right_lower, **dargs) < 0)
    while np.any(invalid):
        right_lower[invalid] += 10 * tol
        invalid = (deriv_logdensityratio(right_lower, **dargs) < 0)

    # Right upper bounds
    invalid = (deriv_logdensityratio(right_upper, **dargs) > 0)
    while np.any(invalid):
        right_upper[invalid] += approx_sd[invalid]
        invalid = (deriv_logdensityratio(right_upper, **dargs) > 0)

    # Find zeros that are less than y_hat using bisection.
    left_roots = vectorized_bisection(f=deriv_logdensityratio, f_kwargs=dargs,
                                      lower=left_lower, upper=left_upper,
                                      tol=tol, maxIter=maxIter)

    # Find zeros that are greater than y_hat using bisection.
    right_roots = vectorized_bisection(f=deriv_logdensityratio, f_kwargs=dargs,
                                       lower=right_lower, upper=right_upper,
                                       tol=tol, maxIter=maxIter)

    # Compute bounding factor M
    f_left_roots = densityratio(left_roots, normalizing_cnst=normalizing_cnst,
                                **dargs)
    f_right_roots = densityratio(
        right_roots, normalizing_cnst=normalizing_cnst,
        **dargs)

    # Store maximum of each root
    M = np.maximum(f_left_roots, f_right_roots)

    # Return results
    return M


def rintensities_cen(n_cen, mu, sigmasq, y_hat, approx_sd,
                     p_int_cen, p_rnd_cen,
                     eta_0, eta_1, propDf,
                     tol=1e-10, maxIter=100):
    '''
    Draw censored intensities and random censoring indicators given nCen and
    quantities computed from Laplace approximation.

    Returns
    -------
        - intensities : ndarray
            A 1d ndarray of sampled censored intensities
        - mapping : ndarray
            A 1d integer ndarray of peptide indices, one per censored state
        - W : ndarray
            A 1d integer ndarray of indicators for random censoring
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
    post_p_rnd_cen = p_rnd_cen / (p_rnd_cen + (1. - p_rnd_cen) * p_int_cen)
    W = (np.random.uniform(
        size=n_states) < post_p_rnd_cen[mapping]).astype(int)

    # Drawing censored intensities
    # First, get the maximum of the target / proposal ratio for each set of
    # unique parameter values (not per state)
    M = bound_density_ratio(eta_0=eta_0, eta_1=eta_1, mu=mu, sigmasq=sigmasq,
                            y_hat=y_hat, approx_sd=approx_sd,
                            normalizing_cnst=1. / p_int_cen, propDf=propDf,
                            tol=tol, maxIter=maxIter)

    # Next, draw randomly-censored intensities
    intensities[W == 1] = np.random.normal(loc=mu[mapping[W == 1]],
                                           scale=np.sqrt(
                                           sigmasq[mapping[W == 1]]),
                                           size=np.sum(W))

    # Draw remaining intensity-censored intensities using rejection sampler
    active = (W == 0)
    while(np.sum(active) > 0):
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
                                   normalizing_cnst=1. /
                                   p_int_cen[mapping[active]],
                                   propDf=propDf, log=False)
        accept_prob /= M[mapping[active]]

        # Accept draws with given probabilities by marking corresponding active
        # entries False.
        u = np.random.uniform(size=np.sum(active))
        active[active] = u > accept_prob

    # Build output
    return (intensities, mapping, W)


#==============================================================================
# General-purpose sampling routines for parallel implementation
#==============================================================================

def balanced_sample(n_items, n_samples):
    '''
    Draw maximally-balanced set of m samples (without replacement) from n items.
    '''
    # Draw sample indices
    s = np.repeat(
        np.arange(n_samples, dtype='i'), np.floor(n_items / n_samples))
    np.random.shuffle(s)
    # Handle stragglers
    stragglers = np.random.permutation(n_samples)[:n_items -
                                                  n_samples * (n_items / n_samples)]

    return np.r_[s, stragglers]

#==============================================================================
# General-purpose MCMC diagnostic and summarization functions
#==============================================================================


def effective_sample_sizes(**kwargs):
    '''
    Estimate effective sample size for each input using AR(1) approximation.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.

    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          effective sample size(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          effective sample size(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    ess = {}

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Demean the draws
        draws = draws - draws.mean(axis=0)

        # Compute lag-1 autocorrelation by column
        acf = np.mean(draws[1:] * draws[:-1], axis=0) / np.var(draws, axis=0)

        # Compute ess from ACF
        ess[var] = np.shape(draws)[0] * (1. - acf) / (1. + acf)

    if len(kwargs) > 1:
        return ess
    else:
        return ess[kwargs.keys()[0]]


def posterior_medians(**kwargs):
    '''
    Estimate posterior medians from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
    - **kwargs
        Names and arrays of MCMC draws.

    Returns
    -------
    - If only one array of draws is provided, a single array containing the
      posterior median estimate(s) for those variables.
    - If multiple arrays are provided, a dictionary with keys identical to
      those provided as parameters and one array per input containing
      posterior median estimate(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    medians = {}

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Estimate posterior means
        medians[var] = np.median(draws, 0)

    if len(kwargs) > 1:
        return medians
    else:
        return medians[kwargs.keys()[0]]


def posterior_means(**kwargs):
    '''
    Estimate posterior means from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.

    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          posterior mean estimate(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          posterior mean estimate(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    means = {}

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Estimate posterior means
        means[var] = np.mean(draws, 0)

    if len(kwargs) > 1:
        return means
    else:
        return means[kwargs.keys()[0]]


def posterior_variances(**kwargs):
    '''
    Estimate posterior variances from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.

    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          posterior variance estimate(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          posterior variance estimate(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    variances = {}

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Estimate posterior means
        variances[var] = np.var(draws, 0)

    if len(kwargs) > 1:
        return variances
    else:
        return variances[kwargs.keys()[0]]


def posterior_stderrors(**kwargs):
    '''
    Estimate posterior standard errors from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.

    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          posterior standard error estimate(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          posterior standard error estimate(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    stderrors = {}

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Estimate posterior means
        stderrors[var] = np.std(draws, 0)

    if len(kwargs) > 1:
        return stderrors
    else:
        return stderrors[kwargs.keys()[0]]


def hpd_intervals(prob=0.95, **kwargs):
    '''
    Estimate HPD intervals from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
    **kwargs
        Names and arrays of MCMC draws.

    Returns
    -------
    - If only one array of draws is provided, a single array containing the
      HPD interval estimate(s) for those variables.
    - If multiple arrays are provided, a dictionary with keys identical to
      those provided as parameters and one array per input containing
      HPD interval estimate(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    intervals = {}

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Estimate HPD intervals, based on HPDinterval function in coda R
        # package
        sorted_draws = np.sort(draws, 0)
        n_draws = draws.shape[0]
        gap = max(1, min(n_draws - 1, int(round(n_draws * prob))))
        inds = [np.argmin(v[gap:] - v[:-gap]) for v in sorted_draws.T]
        hpd = np.empty((draws.shape[1], 2), draws.dtype)
        hpd[:, 0] = [v[i] for v, i in itertools.izip(sorted_draws.T, inds)]
        hpd[:, 1] = [v[i + gap] for v, i in itertools.izip(sorted_draws.T,
                                                           inds)]
        intervals[var] = hpd

    if len(kwargs) > 1:
        return intervals
    else:
        return intervals[kwargs.keys()[0]]


def quantile_intervals(prob=0.95, **kwargs):
    '''
    Estimate quantile-based intervals from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.

    Parameters
    ----------
    **kwargs
        Names and arrays of MCMC draws.

    Returns
    -------
    - If only one array of draws is provided, a single array containing the
      quantile interval estimate(s) for those variables.
    - If multiple arrays are provided, a dictionary with keys identical to
      those provided as parameters and one array per input containing
      quantile interval estimate(s).

    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')

    # Allocate empty dictionary for results
    intervals = {}
    alpha = 1. - prob
    lower_prob = alpha / 2.
    upper_prob = 1. - lower_prob

    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:, np.newaxis]

        # Estimate HPD intervals, based on HPDinterval function in coda R
        # package
        sorted_draws = np.sort(draws, 0)
        n_draws = draws.shape[0]
        lower = int(round(n_draws * lower_prob))
        upper = min(int(round(n_draws * upper_prob)), n_draws - 1)
        qint[:, 0] = [v[lower] for v in sorted_draws.T]
        qint[:, 1] = [v[upper] for v in sorted_draws.T]
        intervals[var] = qint

    if len(kwargs) > 1:
        return intervals
    else:
        return intervals[kwargs.keys()[0]]

#==============================================================================
# General-purpose IO functions
#==============================================================================

def write_to_hdf5(fname, compress='gzip', **kwargs):
    '''
    Write **kwargs to HDF5 file fname, potentially with compression.

    Parameters
    ----------
        - fname : filename
        - compress : string or None
            Compression to use when saving. Can be None, 'gzip', 'lzf', or
            'szip'.
    '''
    # Check input validity
    if type(fname) is not str:
        raise TypeError('fname must be a path to a writable file.')

    if compress not in (None, 'gzip', 'lzf', 'szip'):
        raise ValueError('Invalid value for compress.')

    # Open file
    out_file = h5py.File(fname, 'w')

    # Write other arguments to file
    write_args_to_hdf5(hdf5=out_file, compress=compress, **kwargs)

    # Close file used for output
    out_file.close()


def write_args_to_hdf5(hdf5, compress, **kwargs):
    for k, v in kwargs.iteritems():
        if type(v) is dict or type(v) is h5py.Group:
            d = hdf5.create_group(k)
            write_args_to_hdf5(d, compress, **v)
        else:
            if np.size(v) > 1:
                hdf5.create_dataset(k, data=v, compression=compress)
            else:
                hdf5.create_dataset(k, data=v)


def write_to_pickle(fname, compress='bz2', **kwargs):
    '''
    Pickle **kwargs to fname, potentially with compression.

    Parameters
    ----------
        - fname : filename
        - compress : string or None
            Compression to use when saving. Can be None, 'bz2', or 'gz'.
    '''
    # Check input validity
    if type(fname) is not str:
        raise TypeError('fname must be a path to a writable file.')

    if compress not in (None, 'bz2', 'gz'):
        raise ValueError('Invalid value for compress.')

    # Open file, depending up compression requested
    if compress is None:
        out_file = open(fname, 'wb')
    elif compress == 'bz2':
        out_file = bz2.BZ2File(fname, 'wb')
    elif compress == 'gz':
        out_file = gzip.GzipFile(fname, 'wb')

    # Write other arguments to file
    cPickle.dump(kwargs, out_file, protocol=-1)

    # Close file used for output
    out_file.close()


def convert_dtype_to_fmt(dtype, quote=True):
    '''
    Converts dtype from record array to output format
    Uses %d for integers, %g for floats, and %s for strings
    '''
    # Get kinds
    kinds = [dtype.fields[key][0].kind for key in dtype.names]

    # Iterate through kinds, assigning format as needed
    fmt = []
    for i in range(len(kinds)):
        if kinds[i] in ('b', 'i', 'u'):
            fmt.append('%d')
        elif kinds[i] in ('c', 'f'):
            fmt.append('%g')
        elif kinds[i] in ('S',):
            if quote:
                fmt.append('"%s"')
            else:
                fmt.append('%s')
        else:
            fmt.append('%s')

    return fmt


def write_recarray_to_file(fname, data, header=True, quote=True, sep=' '):
    '''
    Write numpy record array to file as delimited text.

    fname can be either a file name or a file object.

    Works only for numeric data in current form; it will not format strings
    correctly.
    '''
    # Get field names
    fieldnames = data.dtype.names

    # Build header
    if header:
        header_str = sep.join(fieldnames) + '\n'

    # Build format string for numeric data
    fmt = sep.join(convert_dtype_to_fmt(data.dtype, quote)) + '\n'

    # Setup output file object
    if type(fname) is file:
        out_file = fname
    else:
        out_file = open(fname, "wb")

    # Write output
    if header:
        out_file.write(header_str)

    for rec in data:
        out_file.write(fmt % rec.tolist())

    # Close output file
    out_file.close()

