import numpy as np

import lib
import glm

def mcmc_serial(intensities_obs, mapping_states_obs, mapping_peptides, cfg):
    '''
    Serial MCMC sampler for posterior of state-level censoring model.

    Parameters
    ----------
        - intensities_obs : array_like
            A 1d array of length n_obs_states for which each entry contains the
            observed (summed) log state intensity.
            This must be aligned to mapping_states_obs and all entires must be
            > -inf; no missing peptides.
        - mapping_states_obs : array_like, 1 dimension, nonnegative ints
            A 1d integer array of length n_obs_states for which each entry
            contains the index of the peptide that corresponds to the given
            observed state. Peptide indices can range over 0 <= i < n_peptides.
            Not every peptide index is required to appear in this mapping; only
            observed peptides should be included. Also note that peptides are
            indexed overall, not within protein.
        - mapping_peptides : array_like, 1 dimension, nonnegative ints
            A 1d integer array of length n_peptides for which each entry
            contains the index of the protein that corresponds to the given
            peptide. Protein indices can range over 0 <= i < n_proteins.
            Every peptide and protein to be included in the model should be
            included here. That is, both observed and unobserved peptides should
            appear in this mapping.
        - cfg : dictionary
            A dictionary (typically generated from a YAML file) containing
            priors and settings for the MCMC algorithm. Its exact form will be
            documented elsewhere. It will have at least three sections: priors,
            containing one entry per parameter, settings, containing settings
            for the MCMC algorithm, and init, containing initial values for
            certain parameters.

    Returns
    -------
        - out : dictionary of output
            1- and 2-dimensional ndarrays containing the posterior samples for
            each parameter. Exact format and diagnostic information TBD.

    '''
    # Convert inputs to np.ndarrays as needed
    if type(intensities_obs) is not np.ndarray:
        intensities_obs = np.asanyarray(intensities_obs)
    if type(mapping_states_obs) is not np.ndarray:
        mapping_states_obs = np.asanyarray(mapping_states_obs, dtype=np.int)
    if type(mapping_peptides) is not np.ndarray:
        mapping_peptides = np.asanyarray(mapping_peptides, dtype=np.int)

    # Extract dimensions from input

    # Number of iterations from cfg
    n_iterations = cfg['settings']['n_iterations']

    # Number of peptides and proteins from mapping_peptides
    n_peptides = np.size(mapping_peptides)
    n_proteins = 1 + np.max(mapping_peptides)

    # Check for validity of mapping vectors
    if (not issubclass(mapping_states_obs.dtype, np.integer) or
        np.min(mapping_states_obs) < 0 or
        np.max(mapping_states_obs) > n_peptides - 1):
        raise ValueError('State to peptide mapping (mapping_states_obs)'
                         ' is not valid')

    if (not issubclass(mapping_peptides.dtype, np.integer) or
        np.min(mapping_peptides) < 0 or
        np.max(mapping_peptides) > n_peptides - 1):
        raise ValueError('Peptide to protein mapping (mapping_peptides)'
                         ' is not valid')

    # Compute tabulations that are invariant across iterations
    
    # Total number of observed states
    n_obs_states = np.size(intensities_obs)
    
    # Tabulate peptides per protein
    n_peptides_per_protein = np.bincount(mapping_peptides)

    # Tabulate number of observed states per peptide
    n_obs_states_per_peptide = np.bincount(mapping_states_obs,
                                           minlength=n_peptides)

    # Sum observed intensities per peptide
    total_intensity_obs_per_peptide = np.bincount(mapping_states_obs,
                                                  weights=intensities_obs,
                                                  minlength=n_peptides)

    # Allocate data structures for draws

    # Peptide- and protein-level means
    gamma_draws     = np.empty((n_iterations, n_peptides))
    mu_draws        = np.empty((n_iterations, n_proteins))

    # State- and peptide-level variances
    sigmasq_draws   = np.empty((n_iterations, n_proteins))
    tausq_draws     = np.empty((n_iterations, n_proteins))

    # Hyperparameters for state-level variance model
    shape_sigmasq   = np.empty(n_iterations)
    rate_sigmasq    = np.empty(n_iterations)

    # Hyperparameters for peptide-level variance model
    shape_tausq     = np.empty(n_iterations)
    rate_tausq      = np.empty(n_iterations)

    # Censoring probability model parameters
    eta_draws       = np.empty((n_iterations, 2))
    p_rnd_cen       = np.empty(n_iterations)

    # Number of states model parameters
    r               = np.empty(n_iterations)
    lmbda           = np.empty(n_iterations)

    # Compute initial values for MCMC iterations

    # p_rnd_cen from cfg
    p_rnd_cen[0] = cfg['init']['p_rnd_cen']

    # eta from cfg; bivariate normal draw
    with cfg['init']['eta'] as eta0:
        eta_draws[0,0] = eta0['mean'][0] + eta0['sd'][0]*np.random.randn(1)
        eta_draws[0,1] = (eta0['mean'][1] +
                          eta0['cor']*eta0['sd'][1]/eta0['sd'][0]*
                          (eta_draws[0,0] - eta0['mean'][0]))
        eta_draws[0,1] += (np.sqrt(1.-eta0['cor']**2) * eta0['sd'][1] *
                           np.random.randn(1))

    # Number of states parameters from MAP estimator based on number of observed
    # peptides; very crude, but not altogether terrible. Note that this ignores
    # the +1 location shift in the actual n_states distribution.
    r[0], lmbda[0] = lib.map_estimator_nbinom(x=n_obs_states_per_peptide,
                                              **cfg['priors']['n_states_model'])

    # Hyperparameters for state- and peptide-level variance distributions
    # directly from cfg
    shape_sigmasq[0], rate_sigmasq[0]   = (cfg['init']['sigmasq_dist']['shape'],
                                           cfg['init']['sigmasq_dist']['rate'])
    shape_tausq[0], rate_tausq[0]       = (cfg['init']['tausq_dist']['shape'],
                                           cfg['init']['tausq_dist']['rate'])
                                           
    # State- and peptide-level variances via inverse-gamma draws
    sigmasq_draws[0]    = 1./np.random.gamma(shape=shape_sigmasq[0], 
                                             rate=rate_sigmasq[0],
                                             size=n_proteins)
    tausq_draws[0]      = 1./np.random.gamma(shape=shape_tausq[0], 
                                             rate=rate_tausq[0],
                                             size=n_proteins)
    
    # Peptide- and protein-level means using mean observed intensity; excluding
    # missing states and imputing missing peptides as zero
    gamma_draws[0]  = (total_intensity_obs_per_peptide / 
                       np.maximum(1, n_obs_states_per_peptide))
    mu_draws[0]     = (np.bincount(mapping_peptides, gamma_draws[0]) /
                       n_peptides_per_protein)
                       
    # Instantiate GLM family for eta step
    logit_family = glm.families.Binomial(link=glm.links.Logit)
    
    # Initialize dictionary for acceptance statistics
    accept_stats = {'sigmasq_dist' : 0,
                    'tausq_dist' : 0,
                    'n_states_dist' : 0,
                    'eta' : 0}
    
    # Master loop for MCMC iterations
    for t in xrange(1, n_iterations):
        # (1) Draw missing data (n_cen and censored state intensities) given all
        #   other parameters. Exact draw via rejection samplers.

        # (1a) Obtain p_int_cen per peptide and approximatations of censored
        #   intensity posteriors.
        kwargs = {'eta_0' : eta_draws[t-1,0],
                  'eta_1' : eta_draws[t-1,1],
                  'mu' : mu_draws[t-1],
                  'sigmasq' : sigmasq_draws[t-1]}
        cen_dist = lib.characterize_censored_intensity_dist(**kwargs)
        
        # (1b) Draw number of censored states per peptide
        n_cen_states_per_peptide = lib.rncen(n_obs=n_obs_states_per_peptide,
                                             p_rnd_cen=p_rnd_cen[t-1],
                                             p_int_cen=cen_dist['p_int_cen'],
                                             lmbda=lmbda[t-1], r=r[t-1])
        # Update state-level counts
        n_states_per_peptide = (n_obs_states_per_peptide +
                                n_cen_states_per_peptide)
        n_states = np.sum(n_states_per_peptide)
        
        # (1c) Draw censored intensities
        kwargs['n_cen'] = n_cen_states_per_peptide
        kwargs['p_rnd_cen'] = p_rnd_cen[t-1]
        kwargs['propDf'] = cfg['settings']['propDf']
        intensities_cen, mapping_states_cen, W = lib.rintensities_cen(**kwargs)
        
        
        # (2) Update random censoring probability. Gibbs step.
        p_rnd_cen[t] = lib.rgibbs_p_rnd_cen(n_rnd_cen=np.sum(W),
                                            n_states=n_states,
                                            **cfg['priors']['p_rnd_cen'])
        
        
        # Sum observed intensities per peptide
        total_intensity_cen_per_peptide = np.bincount(mapping_states_cen,
                                                      weights=intensities_cen,
                                                      minlength=n_peptides)
        
        # Compute mean intensities per peptide
        mean_intensity_per_peptide = ((total_intensity_obs_per_peptide +
                                       total_intensity_cen_per_peptide) /
                                       n_states_per_peptide)
                                       
                                       
        # (3) Update peptide-level mean parameters (gamma). Gibbs step.
        gamma_draws[t] = lib.rgibbs_gamma(mu=mu_draws[t-1],
                                          tausq=tausq_draws[t-1],
                                          sigmasq=sigmasq_draws[t-1],
                                          y_bar=mean_intensity_per_peptide,
                                          n_states=n_states)
        mean_gamma_by_protein = np.bincount(mapping_peptides,
                                            weights=gamma_draws[t])
        mean_gamma_by_protein /= n_peptides_per_protein
        
        
        # (4) Update protein-level mean parameters (mu). Gibbs step.
        mu_draws[t] = lib.rgibbs_mu(gamma_bar=mean_gamma_by_protein,
                                    tausq=tausq_draws[t-1],
                                    n_peptides=n_peptides,
                                    **cfg['priors']['mu'])
        
        
        # (5) Update state-level variance parameters (sigmasq). Gibbs step.
        rss = np.sum((intensities_obs - gamma_draws[t,mapping_states_obs])**2)
        rss += np.sum((intensities_cen - gamma_draws[t,mapping_states_cen])**2)
        sigmasq_draws[t] = lib.rgibbs_variances(rss=rss, n=n_states,
                                                prior_shape=shape_sigmasq[t-1],
                                                prior_rate=rate_sigmasq[t-1])
        
        
        # (6) Update peptide-level variance parameters (tausq). Gibbs step.
        rss = np.sum((gamma_draws[t] - mu_draws[t,mapping_peptides])**2)
        tausq_draws[t] = lib.rgibbs_variances(rss=rss, n=n_peptides,
                                              prior_shape=shape_tausq[t-1],
                                              prior_rate=rate_tausq[t-1])
        
        
        # (7) Update state-level variance hyperparameters (sigmasq
        #   distribution). Conditional independence-chain MH step.
        result = lib.rmh_variance_hyperparams(variances=sigmasq_draws[t],
                                              shape_prev=shape_sigmasq[t-1],
                                              rate_prev=rate_sigmasq[t-1],
                                              **cfg['priors']['sigmasq_dist'])
        shape_sigmasq[t], rate_sigmasq[t], accept = result
        accept_stats['sigmasq_dist'] += accept
        
        
        # (8) Update peptide-level variance hyperparameterse (tausq
        #   distribution). Conditional independence-chain MH step.
        result = lib.rmh_variance_hyperparams(variances=tausq_draws[t],
                                              shape_prev=shape_tausq[t-1],
                                              rate_prev=rate_tausq[t-1],
                                              **cfg['priors']['tausq_dist'])
        shape_tausq[t], rate_tausq[t], accept = result
        accept_stats['tausq_dist'] += accept
        
        
        # (9) Update parameter for negative-binomial n_states distribution (r
        #   and lmbda). Conditional independence-chain MH step.
        result = lib.rmh_nbinom_hyperparams(x=n_states_per_peptide,
                                            r_prev=r[t-1], p_prev=lmbda[t-1],
                                            **cfg['priors']['n_states_dist'])
        r[t], lmbda[t], accept = result
        accept_stats['n_states_dist'] += accept
        
        
        # (10) Update coefficients of intensity-based probabilistic censoring
        #   model (eta). Conditional independence-chain MH step.
        
        # (10a) Build design matrix and response. Only using observed and
        # intensity-censored states.
        n_at_risk = n_obs_states + np.sum(W)
        X = np.empty((n_at_risk, 2))
        X[:,0] = 1.
        X[:,1] = np.r_[intensities_obs, intensities_cen[W>0]]
        #
        y = np.zeros(n_at_risk)
        y[:n_obs_states] = 1.
        
        # (10b) Estimate GLM parameters.
        fit_eta = glm.glm(y=y, X=X, family=logit_family, info=True)
        
        # (10c) Execute MH step.
        eta_draws[t], accept = glm.mh_update_glm_coef(b_prev=eta_draws[t-1],
                                                      y=y, X=X,
                                                      family=logit_family,
                                                      **fit_eta)
        accept_stats['eta'] += accept
        
    # Build dictionary of draws to return
    draws = {'mu' : mu_draws,
             'gamma' : gamma_draws,
             'eta' : eta_draws,
             'p_rnd_cen' : p_rnd_cen}
    return (draws, accept_stats)
