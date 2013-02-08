import sys

import numpy as np

import lib
import glm
import mcmc_updates_serial as updates

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
        - draws : dictionary
            1- and 2-dimensional ndarrays containing the posterior samples for
            each parameter.
        - accept_stats : dictionary
            Dictionary containing number of acceptances for each MH step.

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
    if (not issubclass(mapping_states_obs.dtype.type, np.integer) or
        np.min(mapping_states_obs) < 0 or
        np.max(mapping_states_obs) > n_peptides - 1):
        raise ValueError('State to peptide mapping (mapping_states_obs)'
                         ' is not valid')

    if (not issubclass(mapping_peptides.dtype.type, np.integer) or
        np.min(mapping_peptides) < 0 or
        np.max(mapping_peptides) > n_peptides - 1):
        raise ValueError('Peptide to protein mapping (mapping_peptides)'
                         ' is not valid')

    # Compute tabulations that are invariant across iterations
    
    # Total number of observed states
    n_obs_states = np.size(intensities_obs)
    
    # Tabulate peptides per protein
    n_peptides_per_protein = np.bincount(mapping_peptides)
    peptides_obs = np.unique(mapping_states_obs)
    n_obs_peptides_per_protein = np.bincount(mapping_peptides[peptides_obs],
                                             minlength=n_proteins)

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
    
    # Number of censored states per peptide
    n_cen_states_per_peptide_draws = np.zeros((n_iterations, n_peptides),
                                              dtype=np.integer)

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
    eta0 = cfg['init']['eta']
    eta_draws[0,0] = eta0['mean'][0] + eta0['sd'][0]*np.random.randn(1)
    eta_draws[0,1] = eta0['mean'][1]
    if eta0['sd'][1] > 0:
        eta_draws[0,1] += (eta0['cor']*eta0['sd'][1]/eta0['sd'][0]*
                          (eta_draws[0,0] - eta0['mean'][0]))
        eta_draws[0,1] += (np.sqrt(1.-eta0['cor']**2) * eta0['sd'][1] *
                           np.random.randn(1))        

    # Number of states parameters from MAP estimator based on number of observed
    # peptides; very crude, but not altogether terrible. Note that this ignores
    # the +1 location shift in the actual n_states distribution.
    kwargs = {'x' : n_obs_states_per_peptide[n_obs_states_per_peptide>0]-1,
              'transform' : True}
    kwargs.update(cfg['priors']['n_states_dist'])
    r[0], lmbda[0] = lib.map_estimator_nbinom(**kwargs)

    # Hyperparameters for state- and peptide-level variance distributions
    # directly from cfg
    shape_sigmasq[0], rate_sigmasq[0]   = (cfg['init']['sigmasq_dist']['shape'],
                                           cfg['init']['sigmasq_dist']['rate'])
    shape_tausq[0], rate_tausq[0]       = (cfg['init']['tausq_dist']['shape'],
                                           cfg['init']['tausq_dist']['rate'])
                                           
    # State- and peptide-level variances via inverse-gamma draws
    sigmasq_draws[0]    = 1./np.random.gamma(shape=shape_sigmasq[0], 
                                             scale=1./rate_sigmasq[0],
                                             size=n_proteins)
    tausq_draws[0]      = 1./np.random.gamma(shape=shape_tausq[0], 
                                             scale=1./rate_tausq[0],
                                             size=n_proteins)
                                             
    # Mapping from protein to peptide conditional variances for convenience
    var_peptide_conditional = sigmasq_draws[0][mapping_peptides]
    
    # Protein-level means using mean observed intensity; excluding missing 
    # peptides
    mu_draws[0]     = (np.bincount(mapping_peptides,
                                   total_intensity_obs_per_peptide /
                                   np.maximum(1,n_obs_states_per_peptide)) /
                       n_obs_peptides_per_protein)
    
    # Peptide-level means using mean observed intensity; imputing missing
    # peptides as protein observed means
    gamma_draws[0] = mu_draws[0][mapping_peptides]
    gamma_draws[0][peptides_obs] = (total_intensity_obs_per_peptide[peptides_obs] / 
                                    n_obs_states_per_peptide[peptides_obs])
                       
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
                  'mu' : gamma_draws[t-1],
                  'sigmasq' : var_peptide_conditional}
        cen_dist = lib.characterize_censored_intensity_dist(**kwargs)
        
        # (1b) Draw number of censored states per peptide
        n_cen_states_per_peptide = lib.rncen(n_obs=n_obs_states_per_peptide,
                                             p_rnd_cen=p_rnd_cen[t-1],
                                             p_int_cen=cen_dist['p_int_cen'],
                                             lmbda=lmbda[t-1], r=r[t-1])
        n_cen_states_per_peptide_draws[t] = n_cen_states_per_peptide
        # Update state-level counts
        n_states_per_peptide = (n_obs_states_per_peptide +
                                n_cen_states_per_peptide)
        n_states_per_protein = np.bincount(mapping_peptides,
                                           weights=n_states_per_peptide)
        n_states = np.sum(n_states_per_peptide)
        
        # (1c) Draw censored intensities
        kwargs['n_cen'] = n_cen_states_per_peptide
        kwargs['p_rnd_cen'] = p_rnd_cen[t-1]
        kwargs['propDf'] = cfg['settings']['propDf']
        kwargs.update(cen_dist)
        intensities_cen, mapping_states_cen, W = lib.rintensities_cen(**kwargs)
        
        
        # (2) Update random censoring probability. Gibbs step.
        p_rnd_cen[t] = updates.rgibbs_p_rnd_cen(n_rnd_cen=np.sum(W),
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
        gamma_draws[t] = updates.rgibbs_gamma(mu=mu_draws[t-1][mapping_peptides],
                                       tausq=tausq_draws[t-1][mapping_peptides],
                                       sigmasq=var_peptide_conditional,
                                       y_bar=mean_intensity_per_peptide,
                                       n_states=n_states_per_peptide)
        mean_gamma_by_protein = np.bincount(mapping_peptides,
                                            weights=gamma_draws[t])
        mean_gamma_by_protein /= n_peptides_per_protein
        
        
        # (4) Update protein-level mean parameters (mu). Gibbs step.
        mu_draws[t] = updates.rgibbs_mu(gamma_bar=mean_gamma_by_protein,
                                    tausq=tausq_draws[t-1],
                                    n_peptides=n_peptides_per_protein,
                                    **cfg['priors']['mu'])
        
        
        # (5) Update state-level variance parameters (sigmasq). Gibbs step.
        rss_by_state = (intensities_obs - gamma_draws[t,mapping_states_obs])**2
        rss_by_protein = np.bincount(mapping_peptides[mapping_states_obs],
                                     weights=rss_by_state,
                                     minlength=n_proteins)
        rss_by_state = (intensities_cen - gamma_draws[t,mapping_states_cen])**2
        rss_by_protein += np.bincount(mapping_peptides[mapping_states_cen],
                                      weights=rss_by_state,
                                      minlength=n_proteins)
        sigmasq_draws[t] = updates.rgibbs_variances(rss=rss_by_protein,
                                                n=n_states_per_protein,
                                                prior_shape=shape_sigmasq[t-1],
                                                prior_rate=rate_sigmasq[t-1])
        
        # Mapping from protein to peptide conditional variances for convenience
        var_peptide_conditional = sigmasq_draws[t][mapping_peptides]
        
        
        # (6) Update peptide-level variance parameters (tausq). Gibbs step.
        rss_by_peptide = (gamma_draws[t] - mu_draws[t,mapping_peptides])**2
        rss_by_protein = np.bincount(mapping_peptides, weights=rss_by_peptide)
        tausq_draws[t] = updates.rgibbs_variances(rss=rss_by_protein,
                                              n=n_peptides_per_protein,
                                              prior_shape=shape_tausq[t-1],
                                              prior_rate=rate_tausq[t-1])
        
        
        # (7) Update state-level variance hyperparameters (sigmasq
        #   distribution). Conditional independence-chain MH step.
        result = updates.rmh_variance_hyperparams(variances=sigmasq_draws[t],
                                              shape_prev=shape_sigmasq[t-1],
                                              rate_prev=rate_sigmasq[t-1],
                                              **cfg['priors']['sigmasq_dist'])
        (shape_sigmasq[t], rate_sigmasq[t]), accept = result
        accept_stats['sigmasq_dist'] += accept
        
        
        # (8) Update peptide-level variance hyperparameters (tausq
        #   distribution). Conditional independence-chain MH step.
        result = updates.rmh_variance_hyperparams(variances=tausq_draws[t],
                                              shape_prev=shape_tausq[t-1],
                                              rate_prev=rate_tausq[t-1],
                                              **cfg['priors']['tausq_dist'])
        (shape_tausq[t], rate_tausq[t]), accept = result
        accept_stats['tausq_dist'] += accept
        
        
        # (9) Update parameter for negative-binomial n_states distribution (r
        #   and lmbda). Conditional independence-chain MH step.
        result = updates.rmh_nbinom_hyperparams(x=n_states_per_peptide-1,
                                            r_prev=r[t-1], p_prev=lmbda[t-1],
                                            **cfg['priors']['n_states_dist'])
        (r[t], lmbda[t]), accept = result
        accept_stats['n_states_dist'] += accept
        
        
        # (10) Update coefficients of intensity-based probabilistic censoring
        #   model (eta). Conditional independence-chain MH step.
        
        # (10a) Build design matrix and response. Only using observed and
        # intensity-censored states.
        n_at_risk = n_obs_states + np.sum(W<1)
        X = np.empty((n_at_risk, 2))
        X[:,0] = 1.
        X[:,1] = np.r_[intensities_obs, intensities_cen[W<1]]
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
        
        if (cfg['settings']['verbose'] > 0 and
            t % cfg['settings']['verbose_interval']==0):
            print >> sys.stderr, 'Iteration %d complete' % t
        
    # Build dictionary of draws to return
    draws = {'mu' : mu_draws,
             'gamma' : gamma_draws,
             'eta' : eta_draws,
             'p_rnd_cen' : p_rnd_cen,
             'lmbda' : lmbda,
             'r' : r,
             'sigmasq' : sigmasq_draws,
             'tausq' : tausq_draws,
             'n_cen_states_per_peptide' : n_cen_states_per_peptide_draws,
             'shape_tausq' : shape_tausq,
             'rate_tausq' : rate_tausq,
             'shape_sigmasq' : shape_sigmasq,
             'rate_sigmasq' : rate_sigmasq}
    return (draws, accept_stats)
