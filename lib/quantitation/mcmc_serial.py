import numpy as np

import lib
import glm

def mcmc_serial(intensities_obs, mapping_states, mapping_peptides, cfg):
    '''
    Serial MCMC sampler for posterior of state-level censoring model.

    Parameters
    ----------
        - intensities_obs : array_like
            A 1d array of length n_obs_states for which each entry contains the
            observed (summed) log state intensity.
            This must be aligned to mapping_states and all entires must be
            > -inf; no missing peptides.
        - mapping_states : array_like, 1 dimension, nonnegative ints
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
    if type(mapping_states) is not np.ndarray:
        mapping_states = np.asanyarray(mapping_states, dtype=np.int)
    if type(mapping_peptides) is not np.ndarray:
        mapping_peptides = np.asanyarray(mapping_peptides, dtype=np.int)

    # Extract dimensionalities from input

    # Number of iterations from cfg
    n_iterations = cfg['settings']['n_iterations']

    # Number of peptides and proteins from mapping_peptides
    n_peptides = np.size(mapping_peptides)
    n_proteins = 1 + np.max(mapping_peptides)

    # Check for validity of mapping vectors
    if (not issubclass(mapping_states.dtype, np.integer) or
        np.min(mapping_states) < 0 or
        np.max(mapping_states) > n_peptides - 1):
        raise ValueError('State to peptide mapping (mapping_states)'
                         ' is not valid')

    if (not issubclass(mapping_peptides.dtype, np.integer) or
        np.min(mapping_peptides) < 0 or
        np.max(mapping_peptides) > n_peptides - 1):
        raise ValueError('Peptide to protein mapping (mapping_peptides)'
                         ' is not valid')

    # Compute tabulations that are invariant across iterations

    # Tabulate peptides per protein
    n_peptides_per_protein = np.bincount(mapping_peptides)

    # Tabulate number of observed states per peptide and per protein
    n_obs_states_per_peptide = np.bincount(mapping_states, minlength=n_peptides)
    n_obs_states_per_protein = np.bincount(mapping_peptides,
                                           weights=n_obs_states_per_peptide)

    # Sum observed intensities per peptide and per protein
    total_intensity_obs_per_peptide = np.bincount(mapping_states,
                                                  weights=intensities_obs,
                                                  minlength=n_peptides)
    total_intensity_obs_per_protein = np.bincount(mapping_peptides,
                                      weights=total_intensity_obs_per_peptide)

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
        eta_draws[0,1] = (eta0['mean'][1] + eta0['cor']*eta0['sd'][1]/eta0['sd'][0]*
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
    
    # Master loop for MCMC iterations
    for t in xrange(1, n_iterations):
        # (1) Draw missing data (n_cen and censored state intensities) given all
        #   other parameters.

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
        n_cen_states_per_protein = np.bincount(mapping_peptides,
                                               weights=n_cen_states_per_peptide)
        n_states_per_peptide = (n_obs_states_per_peptide +
                                n_cen_states_per_peptide)
        n_states_per_protein = (n_obs_states_per_protein +
                                n_cen_states_per_protein)
        
        # (1c) Draw censored intensities
        intensities_cen
    
    # Build dictionary of draws to return
    draws = {'mu' : mu_draws,
             'gamma' : gamma_draws,
             'eta' : eta_draws,
             'p_rnd_cen' : p_rnd_cen}
    return draws
