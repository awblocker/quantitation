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
            documented elsewhere. It will have at least two sections: priors,
            containing one entry per parameter, and settings, containing
            settings for the MCMC algorithm.
            
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
    mu_draws    = np.empty((n_iterations, n_proteins))
    gamma_draws = np.empty((n_iterations, n_peptides))
    
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
    eta         = np.empty((n_iterations, 2))
    p_rnd_cen   = np.empty(n_iterations)
    
    # Number of states model parameters
    r       = np.empty(n_iterations)
    lmbda   = np.empty(n_iterations)
    
    return