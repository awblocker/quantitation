
from lib import *

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
    post_var = 1. / (1. / tausq + n_states / sigmasq)
    post_mean = post_var * (mu / tausq + y_bar / sigmasq * n_states)

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
    post_var = 1. / (n_peptides / tausq + prior_prec)
    post_mean = post_var * (gamma_bar * n_peptides / tausq +
                            prior_mean * prior_prec)

    # Draw mu
    mu = np.random.normal(loc=post_mean, scale=np.sqrt(post_var),
                          size=gamma_bar.size)
    return mu


def rgibbs_variances(rss, n, prior_shape=1., prior_rate=0.):
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
    variances = 1. / np.random.gamma(shape=prior_shape + n / 2.,
                                     scale=1. / (prior_rate + rss / 2.),
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
    p_rnd_cen = np.random.beta(a=n_rnd_cen + prior_a,
                               b=n_states - n_rnd_cen + prior_b)
    return p_rnd_cen


def rmh_variance_hyperparams(variances, shape_prev, rate_prev,
                             prior_mean_log=2.65, prior_prec_log=1. / 0.652 ** 2,
                             prior_shape=1., prior_rate=0.,
                             propDf=5., brent_scale=6., fallback_upper=10000.,
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
    Falls back to profile if information matrix is not (numerically) positive
    definite.

    Returns a 2-tuple consisting of the new (shape, rate) and a boolean
    indicating acceptance.
    '''
    # Compute posterior mode for shape and rate using profile log-posterior
    n = np.size(variances)
    precisions = 1. / variances

    shape_hat, rate_hat = map_estimator_gamma(x=precisions, log=True,
                                              prior_shape=prior_shape,
                                              prior_rate=prior_rate,
                                              prior_mean_log=prior_mean_log,
                                              prior_prec_log=prior_prec_log,
                                              brent_scale=brent_scale,
                                              fallback_upper=fallback_upper)
    if not profile:
        # Propose using a bivariate normal approximate to the joint conditional
        # posterior of (shape, rate)

        # Compute posterior information matrix for parameters
        info = info_posterior_gamma(shape=shape_hat, rate=rate_hat,
                                    x=precisions, log=True,
                                    prior_shape=prior_shape,
                                    prior_rate=prior_rate,
                                    prior_mean_log=prior_mean_log,
                                    prior_prec_log=prior_prec_log)

        # Cholesky decompose information matrix for bivariate draw and
        # density calculations
        try:
            U = linalg.cholesky(info, lower=False)
        except:
            # Fallback to profile draw
            profile = True

        if not profile:
            # Propose shape and rate parameter jointly
            theta_hat = np.log(np.array([shape_hat, rate_hat]))
            z_prop = (np.random.randn(2) /
                      np.sqrt(np.random.gamma(shape=propDf / 2., scale=2.,
                                              size=2) / propDf))
            theta_prop = theta_hat + linalg.solve_triangular(U, z_prop)
            shape_prop, rate_prop = np.exp(theta_prop)

            # Demean and decorrelate previous draws
            theta_prev = np.log(np.array([shape_prev, rate_prev]))
            z_prev = np.dot(U, theta_prev - theta_hat)

            # Compute log-ratio of proposal densities

            # These are transformed bivariate t's with equivalent covariance
            # matrices, so the resulting Jacobian terms cancel. We are left to
            # contend with the z's and the Jacobian terms resulting from
            # exponentiation.
            log_prop_ratio = -np.sum(np.log(1. + z_prop ** 2 / propDf) -
                                     np.log(1. + z_prev ** 2 / propDf))
            log_prop_ratio *= (propDf + 1.) / 2.
            log_prop_ratio += -np.sum(theta_prop - theta_prev)

    if profile:
        # Propose based on profile posterior for shape and exact conditional
        # posterior for rate.

        # Compute proposal variance
        var_prop = 1. / info_profile_posterior_gamma(shape=shape_hat,
                                                     x=precisions, log=True,
                                                     prior_shape=prior_shape,
                                                     prior_rate=prior_rate,
                                                     prior_mean_log=prior_mean_log,
                                                     prior_prec_log=prior_prec_log)

        # Propose shape parameter from log-t
        z_prop = (np.random.randn(1) /
                  np.sqrt(np.random.gamma(shape=propDf / 2., scale=2., size=1) /
                          propDf))
        shape_prop = shape_hat * np.exp(np.sqrt(var_prop) * z_prop)

        # Propose rate parameter given shape from exact gamma conditional
        # posterior
        rate_prop = np.random.gamma(shape=n * shape_prop + prior_shape,
                                    scale=1. / (np.sum(precisions) + prior_rate))

        # Compute log-ratio of proposal densities

        # For proposal, start with log-t proposal for shape
        log_prop_ratio = (dt(shape_prop, mu=np.log(shape_hat),
                             scale=np.sqrt(var_prop), log=True) -
                          dt(shape_prev, mu=np.log(shape_hat),
                             scale=np.sqrt(var_prop), log=True))
        # Then, add conditional gamma proposal for rate
        log_prop_ratio += (
            dgamma(rate_prop, shape=n * shape_prop + prior_shape,
                   rate=np.sum(precisions) + prior_rate,
                   log=True) -
            dgamma(
                rate_prev, shape=n * shape_prop + prior_shape,
                rate=np.sum(precisions) + prior_rate,
                log=True))

    # Compute log-ratio of target densities.
    # This is equivalent for both proposals.

    # For target, start with the likelihood for the precisions
    log_target_ratio = np.sum(dgamma(precisions, shape=shape_prop,
                                     rate=rate_prop, log=True) -
                              dgamma(precisions, shape=shape_prev,
                                     rate=rate_prev, log=True))
    if prior_prec_log > 0:
        # Add the log-normal prior on the shape parameter
        log_target_ratio += (dlnorm(shape_prop, mu=prior_mean_log,
                                    sigmasq=1. / prior_prec_log, log=True) -
                             dlnorm(shape_prev, mu=prior_mean_log,
                                    sigmasq=1. / prior_prec_log, log=True))
    # Add the gamma prior on the rate parameter
    if prior_rate > 0:
        log_target_ratio += (dgamma(rate_prop, shape=prior_shape,
                                    rate=prior_rate, log=True) -
                             dgamma(rate_prev, shape=prior_shape,
                                    rate=prior_rate, log=True))
    else:
        log_target_ratio += np.log(rate_prop / rate_prev) * (shape_prop - 1.)

    # Execute MH update
    return mh_update(
        prop=(shape_prop, rate_prop), prev=(shape_prev, rate_prev),
        log_target_ratio=log_target_ratio,
        log_prop_ratio=log_prop_ratio)


def rmh_nbinom_hyperparams(x, r_prev, p_prev,
                           prior_mean_log=2.65, prior_prec_log=1. / 0.652 ** 2,
                           prior_a=1., prior_b=1.,
                           propDf=5., brent_scale=6., fallback_upper=10000.,
                           profile=False):
    '''
    Metropolis-Hastings steps for negative-binomial hyperparameters given all
    other parameters.

    Using a log-normal prior for the r (convolution) hyperparameter and a
    conditionally-conjugate beta prior for p.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log-transformed.

    Can propose from approximation to joint conditional posterior
    (profile=False) or approximately marginalize over the p parameter (by
    profiling) and propose the p given r exactly (profile=True).
    Falls back to profile if information matrix is not (numerically) positive
    definite.

    Returns a 2-tuple consisting of the new (r, p) and a boolean indicating
    acceptance.
    '''
    # Compute posterior mode for r and p using profile log-posterior
    n = np.size(x)

    r_hat, p_hat = map_estimator_nbinom(x=x, transform=True,
                                        prior_a=prior_a, prior_b=prior_b,
                                        prior_mean_log=prior_mean_log,
                                        prior_prec_log=prior_prec_log,
                                        brent_scale=brent_scale,
                                        fallback_upper=fallback_upper)

    if not profile:
        # Propose using a bivariate normal approximate to the joint conditional
        # posterior of (r, p)

        # Compute posterior information matrix for parameters
        info = info_posterior_nbinom(r=r_hat, p=p_hat, x=x, transform=True,
                                     prior_a=prior_a, prior_b=prior_b,
                                     prior_mean_log=prior_mean_log,
                                     prior_prec_log=prior_prec_log)

        # Cholesky decompose information matrix for bivariate draw and
        # density calculations
        try:
            U = linalg.cholesky(info, lower=False)
        except:
            # Fallback to profile draw
            profile = True

        if not profile:
            # Propose r and p jointly
            theta_hat = np.log(np.array([r_hat, p_hat]))
            theta_hat[1] -= np.log(1. - p_hat)
            z_prop = (np.random.randn(2) /
                      np.sqrt(np.random.gamma(shape=propDf / 2., scale=2.,
                                              size=2) / propDf))
            theta_prop = theta_hat + linalg.solve_triangular(U, z_prop)
            r_prop, p_prop = np.exp(theta_prop)
            p_prop = p_prop / (1. + p_prop)

            # Demean and decorrelate previous draws
            theta_prev = np.log(np.array([r_prev, p_prev]))
            theta_prev[1] -= np.log(1. - p_prev)
            z_prev = np.dot(U, theta_prev - theta_hat)

            # Compute log-ratio of proposal densities

            # These are transformed bivariate t's with equivalent covariance
            # matrices, so the resulting Jacobian terms cancel. We are left to
            # contend with the z's and the Jacobian terms resulting from the
            # exponential and logit transformations.
            log_prop_ratio = -np.sum(np.log(1. + z_prop ** 2 / propDf) -
                                     np.log(1. + z_prev ** 2 / propDf))
            log_prop_ratio *= (propDf + 1.) / 2.
            log_prop_ratio += -(np.log(r_prop) - np.log(r_prev))
            log_prop_ratio += -(np.log(p_prop) + np.log(1. - p_prop)
                                - np.log(p_prev) - np.log(1. - p_prev))

    if profile:
        # Propose based on profile posterior for r and exact conditional
        # posterior for p.

        # Compute proposal variance
        var_prop = 1. / info_profile_posterior_nbinom(r=r_hat, x=x,
                                                      transform=True,
                                                      prior_a=prior_a, prior_b=prior_b,
                                                      prior_mean_log=prior_mean_log,
                                                      prior_prec_log=prior_prec_log)

        # Propose r parameter from log-t
        z_prop = (np.random.randn(1) /
                  np.sqrt(np.random.gamma(shape=propDf / 2., scale=2., size=1) /
                          propDf))
        r_prop = r_hat * np.exp(np.sqrt(var_prop) * z_prop)

        # Propose p parameter given r from exact beta conditional posterior
        p_prop = np.random.beta(a=np.sum(x) + prior_a - 1.,
                                b=n * r_prop + prior_b - 1.)

        # Compute log-ratio of proposal densities

        # For proposal, start with log-t proposal for r
        log_prop_ratio = (dt(r_prop, mu=np.log(r_hat),
                             scale=np.sqrt(var_prop), log=True) -
                          dt(r_prev, mu=np.log(r_hat),
                             scale=np.sqrt(var_prop), log=True))
        # Then, add conditional beta proposal for p
        log_prop_ratio += (dbeta(p_prop, a=np.sum(x) + prior_a - 1.,
                                 b=n * r_prop + prior_b - 1.,
                                 log=True) -
                           dbeta(p_prev, a=np.sum(x) + prior_a - 1.,
                                 b=n * r_prev + prior_b - 1.,
                                 log=True))

    # Compute log-ratio of target densities.
    # This is equivalent for both proposals.

    # For target, start with the likelihood for x
    log_target_ratio = np.sum(dnbinom(x, r=r_prop, p=p_prop, log=True) -
                              dnbinom(x, r=r_prev, p=p_prev, log=True))
    if prior_prec_log > 0:
        # Add the log-normal prior on r
        log_target_ratio += (dlnorm(r_prop, mu=prior_mean_log,
                                    sigmasq=1. / prior_prec_log, log=True) -
                             dlnorm(r_prev, mu=prior_mean_log,
                                    sigmasq=1. / prior_prec_log, log=True))
    # Add the beta prior on p
    log_target_ratio += (dbeta(p_prop, a=prior_a, b=prior_b, log=True) -
                         dbeta(p_prev, a=prior_a, b=prior_b, log=True))

    # Execute MH update
    return mh_update(prop=(r_prop, p_prop), prev=(r_prev, p_prev),
                     log_target_ratio=log_target_ratio,
                     log_prop_ratio=log_prop_ratio)
