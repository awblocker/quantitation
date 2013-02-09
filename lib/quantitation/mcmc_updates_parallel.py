from mpi4py import MPI
from scipy import optimize

import glm

from lib import *
from mcmc_updates_serial import mh_update
from fisher_weighting import *
import emulate

#==============================================================================
# Specialized sampling routines for parallel implementation
#==============================================================================

def rmh_worker_variance_hyperparams(comm, variances, shape_prev, rate_prev,
                                    MPIROOT=0,
                                    prior_mean_log=2.65,
                                    prior_prec_log=1. / 0.652 ** 2,
                                    prior_shape=1., prior_rate=0.,
                                    brent_scale=6., fallback_upper=10000.,
                                    correct_prior=False):
    '''
    Worker side of Metropolis-Hastings step for variance hyperparameters given
    all other parameters.

    Have normal likelihood, so variance likelihood has the same form as gamma
    distribution. Using a log-normal prior for the shape hyperparameter and
    gamma prior for rate hyperparameter.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log-transformed.

    Builds normal approximation based upon local data, then combines this with
    others on the master process. These approximations are used to generate a
    proposal, which is then broadcast back to the workers. The workers then
    evaluate the log-target ratio and combine these on the master to execute
    the MH step. The resulting draw is __not__ brought back to the workers until
    the next synchronization.

    Returns None.
    '''
    # Correct / adjust prior for distributed approximation, if requested
    adj = 1.
    if correct_prior:
        adj = comm.Get_size() - 1.

    # Compute posterior mode for shape and rate using profile log-posterior
    precisions = 1. / variances

    shape_hat, rate_hat = map_estimator_gamma(x=precisions, log=True,
                                              prior_shape=prior_shape,
                                              prior_rate=prior_rate,
                                              prior_mean_log=prior_mean_log,
                                              prior_prec_log=prior_prec_log,
                                              prior_adj=adj,
                                              brent_scale=brent_scale,
                                              fallback_upper=fallback_upper)

    # Propose using a bivariate normal approximate to the joint conditional
    # posterior of (shape, rate)

    # Compute posterior information matrix for parameters
    info = info_posterior_gamma(shape=shape_hat, rate=rate_hat,
                                x=precisions, log=True,
                                prior_shape=prior_shape,
                                prior_rate=prior_rate,
                                prior_mean_log=prior_mean_log,
                                prior_prec_log=prior_prec_log,
                                prior_adj=adj)

    # Compute information-weighted point estimate
    theta_hat = np.log(np.array([shape_hat, rate_hat]))
    z_hat = np.dot(info, theta_hat)

    # Condense approximation to a single vector for reduction
    approx = np.r_[z_hat, info[np.tril_indices(2)]]

    # Combine with other approximations on master.
    comm.Reduce([approx, MPI.DOUBLE], None,
                op=MPI.SUM, root=MPIROOT)

    # Obtain proposed value of theta from master.
    theta_prop = np.empty(2)
    comm.Bcast([theta_prop, MPI.DOUBLE], root=MPIROOT)
    shape_prop, rate_prop = np.exp(theta_prop)

    # Compute log-ratio of target densities, omitting prior.
    # Log-ratio of prior densities is handled on the master.

    # Only component is the likelihood for the precisions
    log_target_ratio = np.sum(dgamma(precisions, shape=shape_prop,
                                     rate=rate_prop, log=True) -
                              dgamma(precisions, shape=shape_prev,
                                     rate=rate_prev, log=True))

    # Reduce log-target ratio for MH step on master.
    comm.Reduce([np.array(log_target_ratio), MPI.DOUBLE], None,
                op=MPI.SUM, root=MPIROOT)

    # All subsequent computation is handled on the master node.
    # Synchronization of the resulting draw is handled separately.


def rmh_master_variance_hyperparams(comm, shape_prev, rate_prev, MPIROOT=0,
                                    prior_mean_log=2.65,
                                    prior_prec_log=1. / 0.652 ** 2,
                                    prior_shape=1., prior_rate=0.,
                                    propDf=5.):
    '''
    Master side of Metropolis-Hastings step for variance hyperparameters given
    all other parameters.

    Have normal likelihood, so variance likelihood has the same form as gamma
    distribution. Using a log-normal prior for the shape hyperparameter and
    gamma prior for rate hyperparameter.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log-transformed.

    Builds normal approximation based upon local data, then combines this with
    others on the master process. These approximations are used to generate a
    proposal, which is then broadcast back to the workers. The workers then
    evaluate the log-target ratio and combine these on the master to execute
    the MH step. The resulting draw is __not__ brought back to the workers until
    the next synchronization.

    Returns a 2-tuple consisting of the new (shape, rate) and a boolean
    indicating acceptance.
    '''
    # Build normal approximation to posterior of transformed hyperparameters.
    # Aggregating local results from workers.
    # This assumes that rmh_worker_nbinom_hyperparams() has been called on all
    # workers.
    theta_hat, prec = posterior_approx_distributed(comm=comm, dim_param=2,
                                                   MPIROOT=MPIROOT)

    # Cholesky decompose information matrix for bivariate draw and
    # density calculations
    U = linalg.cholesky(prec, lower=False)

    # Propose shape and rate parameter jointly
    z_prop = (np.random.randn(2) /
              np.sqrt(np.random.gamma(shape=propDf / 2., scale=2.,
                                      size=2) / propDf))
    theta_prop = theta_hat + linalg.solve_triangular(U, z_prop)
    shape_prop, rate_prop = np.exp(theta_prop)

    # Demean and decorrelate previous draws
    theta_prev = np.log(np.array([shape_prev, rate_prev]))
    z_prev = np.dot(U, theta_prev - theta_hat)

    # Broadcast theta_prop to workers
    comm.Bcast([theta_prop, MPI.DOUBLE], root=MPIROOT)

    # Compute log-ratio of target densities.
    # Start by obtaining likelihood component from workers.
    log_target_ratio = np.array(0.)
    buf = np.array(0.)
    comm.Reduce([buf, MPI.DOUBLE], [log_target_ratio, MPI.DOUBLE],
                op=MPI.SUM, root=MPIROOT)

    # Add log-prior ratio
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

    # Compute log-ratio of proposal densities

    # These are transformed bivariate t's with equivalent covariance
    # matrices, so the resulting Jacobian terms cancel. We are left to
    # contend with the z's and the Jacobian terms resulting from
    # exponentiation.
    log_prop_ratio = -np.sum(np.log(1. + z_prop ** 2 / propDf) -
                             np.log(1. + z_prev ** 2 / propDf))
    log_prop_ratio *= (propDf + 1.) / 2.
    log_prop_ratio += -np.sum(theta_prop - theta_prev)

    # Execute MH update
    return mh_update(
        prop=(shape_prop, rate_prop), prev=(shape_prev, rate_prev),
        log_target_ratio=log_target_ratio,
        log_prop_ratio=log_prop_ratio)


def rmh_worker_nbinom_hyperparams(comm, x, r_prev, p_prev, MPIROOT=0,
                                  prior_mean_log=2.65,
                                  prior_prec_log=1. / 0.652 ** 2,
                                  prior_a=1., prior_b=1.,
                                  brent_scale=6., fallback_upper=10000.,
                                  correct_prior=True):
    '''
    Worker side of Metropolis-Hastings step for negative-binomial
    hyperparameters given all other parameters.

    Using a log-normal prior for the r (convolution) hyperparameter and a
    conditionally-conjugate beta prior for p.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log- and logit-transformed.

    Builds normal approximation based upon local data, then combines this with
    others on the master process. These approximations are used to generate a
    proposal, which is then broadcast back to the workers. The workers then
    evaluate the log-target ratio and combine these on the master to execute
    the MH step. The resulting draw is __not__ brought back to the workers until
    the next synchronization.

    Returns None.
    '''
    # Correct / adjust prior for distributed approximation, if requested
    adj = 1.
    if correct_prior:
        adj = comm.Get_size() - 1.

    # Compute posterior mode for r and p using profile log-posterior
    r_hat, p_hat = map_estimator_nbinom(x=x, transform=True,
                                        prior_a=prior_a, prior_b=prior_b,
                                        prior_mean_log=prior_mean_log,
                                        prior_prec_log=prior_prec_log,
                                        prior_adj=adj,
                                        brent_scale=brent_scale,
                                        fallback_upper=fallback_upper)

    # Propose using a bivariate normal approximate to the joint conditional
    # posterior of (r, p)

    # Compute posterior information matrix for parameters
    info = info_posterior_nbinom(r=r_hat, p=p_hat, x=x, transform=True,
                                 prior_a=prior_a, prior_b=prior_b,
                                 prior_mean_log=prior_mean_log,
                                 prior_prec_log=prior_prec_log, prior_adj=adj)

    # Compute information-weighted point estimate
    theta_hat = np.log(np.array([r_hat, p_hat]))
    theta_hat[1] -= np.log(1. - p_hat)
    z_hat = np.dot(info, theta_hat)

    # Condense approximation to a single vector for reduction
    approx = np.r_[z_hat, info[np.tril_indices(2)]]

    # Combine with other approximations on master.
    comm.Reduce([approx, MPI.DOUBLE], None,
                op=MPI.SUM, root=MPIROOT)

    # Obtain proposed value of theta from master.
    theta_prop = np.empty(2)
    comm.Bcast([theta_prop, MPI.DOUBLE], root=MPIROOT)
    r_prop, p_prop = np.exp(theta_prop)
    p_prop = p_prop / (1. + p_prop)

    # Compute log-ratio of target densities, omitting prior.
    # Log-ratio of prior densities is handled on the master.

    # Only component is log-likelihood ratio for x.
    log_target_ratio = np.sum(dnbinom(x, r=r_prop, p=p_prop, log=True) -
                              dnbinom(x, r=r_prev, p=p_prev, log=True))

    # Reduce log-target ratio for MH step on master.
    comm.Reduce([np.array(log_target_ratio), MPI.DOUBLE], None,
                op=MPI.SUM, root=MPIROOT)

    # All subsequent computation is handled on the master node.
    # Synchronization of the resulting draw is handled separately.


def rmh_master_nbinom_hyperparams(comm, r_prev, p_prev, MPIROOT=0,
                                  prior_mean_log=2.65,
                                  prior_prec_log=1. / 0.652 ** 2,
                                  prior_a=1., prior_b=1.,
                                  propDf=5.):
    '''
    Master side of Metropolis-Hastings step for negative-binomial
    hyperparameters given all other parameters.

    Using a log-normal prior for the r (convolution) hyperparameter and a
    conditionally-conjugate beta prior for p.

    Proposing from normal approximation to the conditional posterior
    (conditional independence chain). Parameters are log- and logit-transformed.

    Builds normal approximation based upon local data, then combines this with
    others on the master process. These approximations are used to generate a
    proposal, which is then broadcast back to the workers. The workers then
    evaluate the log-target ratio and combine these on the master to execute
    the MH step. The resulting draw is __not__ brought back to the workers until
    the next synchronization.

    Returns a 2-tuple consisting of the new (shape, rate) and a boolean
    indicating acceptance.
    '''
    # Build normal approximation to posterior of transformed hyperparameters.
    # Aggregating local results from workers.
    # This assumes that rmh_worker_nbinom_hyperparams() has been called on all
    # workers.
    theta_hat, prec = posterior_approx_distributed(comm=comm, dim_param=2,
                                                   MPIROOT=MPIROOT)

    # Cholesky decompose information matrix for bivariate draw and
    # density calculations
    U = linalg.cholesky(prec, lower=False)

    # Propose r and p jointly
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

    # Broadcast theta_prop to workers
    comm.Bcast([theta_prop, MPI.DOUBLE], root=MPIROOT)

    # Compute log-ratio of target densities.
    # Start by obtaining likelihood component from workers.
    log_target_ratio = np.array(0.)
    buf = np.array(0.)
    comm.Reduce([buf, MPI.DOUBLE], [log_target_ratio, MPI.DOUBLE],
                op=MPI.SUM, root=MPIROOT)

    if prior_prec_log > 0:
        # Add the log-normal prior on r
        log_target_ratio += (dlnorm(r_prop, mu=prior_mean_log,
                                    sigmasq=1. / prior_prec_log, log=True) -
                             dlnorm(r_prev, mu=prior_mean_log,
                                    sigmasq=1. / prior_prec_log, log=True))
    # Add the beta prior on p
    log_target_ratio += (dbeta(p_prop, a=prior_a, b=prior_b, log=True) -
                         dbeta(p_prev, a=prior_a, b=prior_b, log=True))

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

    # Execute MH update
    return mh_update(prop=(r_prop, p_prop), prev=(r_prev, p_prev),
                     log_target_ratio=log_target_ratio,
                     log_prop_ratio=log_prop_ratio)


def rmh_worker_glm_coef(comm, b_hat, b_prev, y, X, I, family, w=1, V=None,
                        method='emulate', MPIROOT=0, coverage_prob=0.999,
                        grid_min_spacing=0.5, cov=emulate.cov_sqexp, **kwargs):
    '''
    Worker component of single Metropolis-Hastings step for GLM coefficients
    using a normal approximation to their posterior distribution. Proposes
    linearly-transformed vector of independent t_propDf random variables.

    At least one of I (the Fisher information) and V (the inverse Fisher
    information) must be provided. If I is provided, V is ignored. It is more
    efficient to provide the information matrix than the covariance matrix.

    Returns None.
    '''
    # Get number of workers
    n_workers = comm.Get_size() - 1

    # Get dimensions
    p = X.shape[1]
    
    if method == 'emulate':
        # Build local emulator for score function
        grid_radius = emulate.approx_quantile(coverage_prob=coverage_prob,
                                              d=p, n=n_workers)
        if V is None:
            L = linalg.solve_triangular(linalg.cholesky(I, lower=True),
                                        np.eye(p), lower=True)
        else:
            L = linalg.cholesky(V, lower=True)

        emulator = emulate.build_emulator(
            glm.score, center=b_hat, slope_mean=L, cov=cov,
            grid_min_spacing=grid_min_spacing, grid_radius=grid_radius,
            f_kwargs={'y' : y, 'X' : X, 'family' : family})
        
        # Send emulator to master node
        emulate.aggregate_emulators_mpi(
            comm=comm, emulator=emulator, MPIROOT=MPIROOT)
    else:
        # Build necessary quantities for distributed posterior approximation
        z_hat = np.dot(I, b_hat)

        # Condense approximation to a single vector for reduction
        approx = np.r_[z_hat, I[np.tril_indices(2)]]

        # Combine with other approximations on master.
        comm.Reduce([approx, MPI.DOUBLE], None,
                    op=MPI.SUM, root=MPIROOT)

        # Receive settings for refinement
        settings = np.zeros(2, dtype=int)
        comm.Bcast([settings, MPI.INT], root=MPIROOT)
        n_iter, final_info = settings

        # Newton-Raphson iterations for refinement of approximation
        for i in xrange(n_iter):
            # Receive updated estimate from master
            comm.Bcast([b_hat, MPI.DOUBLE], root=MPIROOT)

            # Compute score and information matrix at combined estimate
            eta = np.dot(X, b_hat)
            mu = family.link.inv(eta)
            weights = w * family.weights(mu)
            dmu_deta = family.link.deriv(eta)
            sqrt_W_X = (X.T * np.sqrt(weights)).T

            grad = np.dot(X.T, weights / dmu_deta * (y - mu))
            info = np.dot(sqrt_W_X.T, sqrt_W_X)

            # Condense update to a single vector for reduction
            update = np.r_[grad, info[np.tril_indices(2)]]

            # Combine with other updates on master
            comm.Reduce([update, MPI.DOUBLE], None,
                        op=MPI.SUM, root=MPIROOT)

        # Contribute to final information matrix refinement if requested
        if final_info:
            # Receive updated estimate
            comm.Bcast([b_hat, MPI.DOUBLE], root=MPIROOT)

            # Update information matrix
            eta = np.dot(X, b_hat)
            mu = family.link.inv(eta)
            weights = w * family.weights(mu)
            sqrt_W_X = (X.T * np.sqrt(weights)).T

            info = np.dot(sqrt_W_X.T, sqrt_W_X)

            # Combine informations on master
            comm.Reduce([info[np.tril_indices(2)], MPI.DOUBLE], None,
                        op=MPI.SUM, root=MPIROOT)

    # Obtain proposed value of coefficients from master.
    b_prop = np.empty(p)
    comm.Bcast([b_prop, MPI.DOUBLE], root=MPIROOT)

    # Compute proposed and previous means
    eta_prop = np.dot(X, b_prop)
    eta_prev = np.dot(X, b_prev)

    mu_prop = family.link.inv(eta_prop)
    mu_prev = family.link.inv(eta_prev)

    # Compute log-ratio of target densities
    log_target_ratio = np.sum(family.loglik(y=y, mu=mu_prop, w=w) -
                              family.loglik(y=y, mu=mu_prev, w=w))

    # Reduce log-target ratio for MH step on master.
    comm.Reduce([np.array(log_target_ratio), MPI.DOUBLE], None,
                op=MPI.SUM, root=MPIROOT)

    # All subsequent computation is handled on the master node.
    # Synchronization of the resulting draw is handled separately.


def rmh_master_glm_coef(comm, b_prev, MPIROOT=0., propDf=5., method='emulate',
                        cov=emulate.cov_sqexp, n_iter_refine=2,
                        final_info_refine=1):
    '''
    Master component of single Metropolis-Hastings step for GLM coefficients
    using a normal approximation to their posterior distribution. Proposes
    linearly-transformed vector of independent t_propDf random variables.

    Builds normal approximation based upon local data, then combines this with
    others on the master process. These approximations are used to generate a
    proposal, which is then broadcast back to the workers. The workers then
    evaluate the log-target ratio and combine these on the master to execute
    the MH step. The resulting draw is __not__ brought back to the workers until
    the next synchronization.

    Returns a 2-tuple consisting of the resulting coefficients and a boolean
    indicating acceptance.
    '''
    # Compute dimensions
    p = np.size(b_prev)

    if method=='emulate':
        # Gather emulators from workers
        emulator = emulate.aggregate_emulators_mpi(
            comm=comm, emulator=None, MPIROOT=0)

        # Find root of combined approximate score function
        b_hat = emulator['center']
        b_hat = optimize.fsolve(
            func=emulate.evaluate_emulator, x0=b_hat, args=(emulator, cov))

        # Compute Cholesky decomposition of approximate combined information
        # for proposal
        U = linalg.solve_triangular(emulator['slope_mean'], np.eye(p),
                                    lower=True).T
    else:
        # Build normal approximation to posterior of transformed hyperparameters.
        # Aggregating local results from workers.
        # This assumes that rmh_worker_nbinom_glm_coef() has been called on all
        # workers.
        b_hat, prec = posterior_approx_distributed(
            comm=comm, dim_param=p, MPIROOT=MPIROOT)

        # Refine approximation with single Newton-Raphson step
        b_hat, prec = refine_distributed_approx(
            comm=comm, est=b_hat, prec=prec, dim_param=p, n_iter=n_iter_refine,
            final_info=final_info_refine, MPIROOT=MPIROOT)

        # Cholesky decompose precision matrix for draws and density calculations
        U = linalg.cholesky(prec, lower=False)

    # Propose from linearly-transformed t with appropriate mean and covariance
    z_prop = (np.random.randn(p) /
              np.sqrt(np.random.gamma(shape=propDf / 2., scale=2., size=p) /
                      propDf))
    b_prop = b_hat + linalg.solve_triangular(U, z_prop, lower=False)

    # Demean and decorrelate previous draw of b
    z_prev = np.dot(U, b_prev - b_hat)

    # Broadcast b_prop to workers
    comm.Bcast([b_prop, MPI.DOUBLE], root=MPIROOT)

    # Compute log-ratio of target densities.
    # Start by obtaining likelihood component from workers.
    log_target_ratio = np.array(0.)
    buf = np.array(0.)
    comm.Reduce([buf, MPI.DOUBLE], [log_target_ratio, MPI.DOUBLE],
                op=MPI.SUM, root=MPIROOT)

    # Compute log-ratio of proposal densities. This is very easy with the
    # demeaned and decorrelated values z.
    log_prop_ratio = -(propDf + 1.) / 2. * np.sum(np.log(1. + z_prop ** 2 / propDf) -
                                                  np.log(1. + z_prev ** 2 / propDf))

    return mh_update(prop=b_prop, prev=b_prev,
                     log_target_ratio=log_target_ratio,
                     log_prop_ratio=log_prop_ratio)


def rgibbs_worker_p_rnd_cen(comm, n_rnd_cen, n_states, MPIROOT=0):
    '''
    Worker component of Gibbs update for p_rnd_cen given all other parameters.

    This is a conjugate beta draw with Bernoulli observations.
    n_rnd_cen must be an integer with the total number of randomly-censored
    states.
    n_states must be an integer with the total number of states (including
    imputed).
    '''
    # Combine counts with other workers on master
    n = np.array([n_rnd_cen, n_states], dtype=np.int)
    comm.Reduce([n, MPI.INT], None,
                op=MPI.SUM, root=MPIROOT)

    # All subsequent computation is handled on the master node.
    # Synchronization of the resulting draw is handled separately.


def rgibbs_master_p_rnd_cen(comm, MPIROOT=0, prior_a=1., prior_b=1.):
    '''
    Master component of Gibbs update for p_rnd_cen given all other parameters.

    This is a conjugate beta draw with Bernoulli observations.
    prior_a and prior_b are the parameters of a conjugate beta prior.
    '''
    # Collect counts of randomly censored and total states from workers
    n = np.empty(2, dtype=np.int)
    buf = np.zeros(2, dtype=np.int)
    comm.Reduce([buf, MPI.INT], [n, MPI.INT],
                op=MPI.SUM, root=MPIROOT)

    n_rnd_cen = n[0]
    n_states = n[1]

    p_rnd_cen = np.random.beta(a=n_rnd_cen + prior_a,
                               b=n_states - n_rnd_cen + prior_b)
    return p_rnd_cen
