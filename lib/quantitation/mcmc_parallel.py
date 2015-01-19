import sys

import numpy as np
from mpi4py import MPI

from quantitation import lib
import glm
import quantitation.mcmc_updates_serial as updates_serial
import quantitation.mcmc_updates_parallel as updates_parallel

# Set constants

# MPI constants
MPIROOT = 0

# Tags for worker states
TAGS = ['STOP',      # Stop iterations
        'INIT',      # Initialize before first iteration
        'SYNC',      # Synchronize hyperparameter values
        'LOCAL',     # Run local, protein-specific draws
        'PRNDCEN',   # Run distributed Gibbs update on p_rnd_cen
        'TAU',       # Run distributed MH update on tausq hyperparams
        'SIGMA',     # Run distributed MH update on sigmasq hyperparams
        'NSTATES',   # Run distributed MH update on n_states hyperparams
        'ETA',       # Run distributed MH update on eta (censoring coef)
        'BETA',      # Run distributed Gibbs update on beta hyperparams
        'CONCENTRATION_DIST',  # Run distributed Gibbs update on concentration
                               # hyperparameters
        'SAVE'       # Save results
       ]
TAGS = dict([(state, tag) for tag, state in enumerate(TAGS)])


def load_data(cfg, rank=None, n_workers=None):
    '''
    Load and setup data based upon process's rank. If rank is None, this loads
    all of the data. If rank is 0, loads only the peptide to protein mapping.
    If rank is > 0, loads all data then reduces to a subset of proteins via
    stratified sampling. Strata are defined by mean observed intensity.
    Number of strata and random seed for sampling are defined in the settings
    section of cfg. The numpy RNG state is restored after this process.

    Parameters
    ----------
        - cfg : dictionary
            Dictionary containing (at least) data section with paths
            to template, null, read data, and regions. It should also have
            a sampling_seed key in its settings section if rank > 0.
        - rank : integer
            Rank of process to load data into.
        - n_workers : integer
            Number of workers to distribute data across. MUST BE SPECIFIED if
            rank > 0.

    Returns
    -------
        Dictionary containing
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
        - proteins_worker : array_like, 1 dimension, nonnegative ints
            A 1d integer array of length n_proteins containing the indices (in
            the original dataset) of the proteins assigned to the given worker.
            Only provided if rank > 0.
        - peptides_worker : array_like, 1 dimension, nonnegative ints
            A 1d integer array of length n_peptides containing the indices (in
            the original dataset) of the peptides assigned to the given worker.
            Only provided if rank > 0.
    '''
    # Determine whether algorithm is running with supervision
    try:
        supervised = cfg['priors']['supervised']
    except KeyError:
        print >> sys.stderr, 'Defaulting to unsupervised algorithm'
        supervised = False

    # Determine whether features are provided
    have_peptide_features = cfg['data'].has_key('path_peptide_features')

    # Load peptide to protein mapping
    mapping_peptides = np.loadtxt(cfg['data']['path_mapping_peptides'],
                                  dtype=np.int)
    n_peptides = np.size(mapping_peptides)

    # Check for validity of peptide to protein mapping
    if (not issubclass(mapping_peptides.dtype.type, np.integer) or
            np.min(mapping_peptides) < 0 or
            np.max(mapping_peptides) > n_peptides - 1):
        raise ValueError('Peptide to protein mapping (mapping_peptides)'
                         ' is not valid')

    # Load peptide-level features, if supplied
    if have_peptide_features:
        data_peptide_features = np.loadtxt(
            cfg['data']['path_peptide_features'], ndmin=2)

    if rank == 0:
        # Return just the peptide to protein mapping if this is the root
        data = {'intensities_obs': None,
                'mapping_states_obs': None,
                'mapping_peptides': mapping_peptides}
        if have_peptide_features:
            data.update({'dim_peptide_features': data_peptide_features.shape})
        return data

    # For everyone else, load the state-level data
    path_data_state = cfg['data']['path_data_state']
    mapping_states_obs, intensities_obs = np.loadtxt(
        path_data_state,
        dtype=[('peptide', np.int), ('intensity', np.float)],
        unpack=True)

    # Check for validity of state to peptide mapping
    if (not issubclass(mapping_states_obs.dtype.type, np.integer) or
            np.min(mapping_states_obs) < 0 or
            np.max(mapping_states_obs) > n_peptides - 1):
        raise ValueError('State to peptide mapping (mapping_states_obs)'
                         ' is not valid')

    # Load concentration data if supervised
    if supervised:
        data_concentrations = np.loadtxt(
            cfg['data']['path_concentrations'],
            dtype=[('protein', np.int),
                   ('concentration', np.float)],
            unpack=True)
        mapping_known_concentrations, known_concentrations = data_concentrations

    if rank is None:
        # Return everything
        data = {'intensities_obs': intensities_obs,
                'mapping_states_obs': mapping_states_obs,
                'mapping_peptides': mapping_peptides}
        if supervised:
            data.update({
                'known_concentrations': known_concentrations,
                'mapping_known_concentrations': mapping_known_concentrations})
        return data

    # Compute observed mean intensity by protein
    n_proteins = np.max(mapping_peptides) + 1
    n_obs_states_per_protein = np.bincount(
        mapping_peptides[mapping_states_obs],
        minlength=n_proteins)
    intensities_obs_mean = np.bincount(mapping_peptides[mapping_states_obs],
                                       weights=intensities_obs,
                                       minlength=n_proteins)
    intensities_obs_mean /= np.maximum(n_obs_states_per_protein, 1)

    # Rank and bin observed mean intensities to create strata
    n_strata = cfg['settings']['n_strata']
    protein_ranks = np.argsort(intensities_obs_mean)

    n_proteins_per_stratum, bins = np.histogram(protein_ranks, bins=n_strata)
    strata = np.digitize(protein_ranks, bins) - 1
    strata[strata < 0] = 0
    strata[strata == n_strata] = n_strata - 1

    # Compute balanced sample of strata, caching and restoring the RNG state
    rng_state = np.random.get_state()
    np.random.seed(cfg['settings']['seed_load_data'])

    workers_by_protein = np.empty(n_proteins, dtype=np.int64)
    for stratum in xrange(n_strata):
        workers = lib.balanced_sample(np.sum(strata == stratum), n_workers)
        workers_by_protein[strata == stratum] = workers

    np.random.set_state(rng_state)

    # Subset data for individual worker
    workers_by_peptide = workers_by_protein[mapping_peptides]
    workers_by_state = workers_by_peptide[mapping_states_obs]

    # Subset intensities
    intensities_obs_worker = intensities_obs[workers_by_state == (rank - 1)]

    # Subset features, if available
    if have_peptide_features:
        peptide_features_worker = data_peptide_features[workers_by_peptide ==
                                                        (rank - 1)]

    # Subset and reindex peptide to protein mapping
    select_peptides = np.where(workers_by_peptide == (rank - 1))[0]
    unique_proteins, mapping_peptides_worker = np.unique(
        mapping_peptides[
            select_peptides],
        return_index=False,
        return_inverse=True)

    # Subset and reindex state to peptide mapping
    mask_states = (workers_by_state == (rank - 1))
    mapping_states_obs_worker = np.empty(np.sum(mask_states),
                                         dtype=np.int)
    reindex = dict(zip(select_peptides, range(select_peptides.size)))
    for peptide_original, peptide_worker in reindex.iteritems():
        mask_edit = (mapping_states_obs[mask_states] == peptide_original)
        mapping_states_obs_worker[mask_edit] = peptide_worker

    if supervised:
        # Subset and reindex known concentrations
        known_concentrations_in_sample = np.where(np.in1d(
            mapping_known_concentrations, unique_proteins,
            assume_unique=True))[0]
        known_concentrations_worker = known_concentrations[
            known_concentrations_in_sample]

        mapping_known_concentrations_worker = np.empty(
            known_concentrations_worker.size, dtype=np.int)
        reindex = zip(
            mapping_known_concentrations[known_concentrations_in_sample],
            range(known_concentrations_worker.size))
        for protein_original, protein_worker in reindex:
            mapping_known_concentrations_worker[protein_worker] = \
                    np.where(unique_proteins == protein_original)[0][0]

    # Build dictionary of data to return
    data = {'intensities_obs': intensities_obs_worker,
            'mapping_states_obs': mapping_states_obs_worker,
            'mapping_peptides': mapping_peptides_worker,
            'proteins_worker': unique_proteins,
            'peptides_worker': select_peptides}
    if supervised:
        data.update({
            'known_concentrations': known_concentrations_worker,
            'mapping_known_concentrations': mapping_known_concentrations_worker}
        )
    if have_peptide_features:
        data.update({
            'peptide_features_worker': peptide_features_worker
        })
    return data


def master(comm, data, cfg):
    '''
    Master node process for parallel MCMC. Coordinates draws and collects
    results.

    Parameters
    ----------
        - comm : mpi4py.MPI.COMM
            Initialized MPI communicator.
        - data : dictionary
            Data as output from load_data with rank==0.
        - cfg : dictionary
            Configuration dictionary containing priors, settings, and paths for
            analysis. Its format is specified in detail in separate
            documentation.

    Returns
    -------
        - draws : dictionary
            1- and 2-dimensional ndarrays containing the posterior samples for
            each shared parameter. Protein- and peptide-specific parameters are
            handled by each worker.
        - accept_stats : dictionary
            Dictionary containing number of acceptances for each MH step.
        - mapping_peptides : integer ndarray
            Peptide to protein mapping provided in data. This is useful for
            merging worker-level results.

    '''
    # Determine number of workers
    n_workers = comm.Get_size() - 1
    if n_workers < 1:
        raise ValueError('Need at least one worker')

    # Determine whether algorithm is running with supervision
    try:
        supervised = cfg['priors']['supervised']
    except KeyError:
        print >> sys.stderr, 'Defaulting to unsupervised algorithm'
        supervised = False

    if 'dim_peptide_features' in data:
        n_peptide_features = data['dim_peptide_features'][1]
    else:
        n_peptide_features = 0

    # If supervised, determine whether to model distribution of concentrations
    # If this is False, prior on $\beta_1$ is scaled by $|\beta_1|^{n_{mis}}$.
    if supervised:
        try:
            concentration_dist = cfg['priors']['concentration_dist']
        except KeyError:
            print >> sys.stderr, 'Defaulting to flat prior on concentrations'
            concentration_dist = False

    # Extract proposal DFs
    try:
        prop_df_eta = cfg['settings']['prop_df_eta']
    except KeyError:
        prop_df_eta = 10.

    # Extract number of iterations from cfg
    n_iterations = cfg['settings']['n_iterations']

    # Allocate data structures for draws

    # Hyperparameters for state-level variance model
    shape_sigmasq = np.empty(n_iterations)
    rate_sigmasq = np.empty(n_iterations)

    # Hyperparameters for peptide-level variance model
    shape_tausq = np.empty(n_iterations)
    rate_tausq = np.empty(n_iterations)

    # Censoring probability model parameters
    eta_draws = np.zeros((n_iterations, 2 + n_peptide_features * 2))
    p_rnd_cen = np.empty(n_iterations)

    # Number of states model parameters
    r = np.empty(n_iterations)
    lmbda = np.empty(n_iterations)

    if supervised:
        # Coefficients for concentration-intensity relationship
        beta_draws = np.zeros((n_iterations, 2))
        # Hyperparameters for concentration distribution. Must be initialized to
        # zero.
        mean_concentration_draws = np.zeros(n_iterations)
        prec_concentration_draws = np.zeros(n_iterations)

    # Compute initial values for MCMC iterations

    # p_rnd_cen from cfg
    p_rnd_cen[0] = cfg['init']['p_rnd_cen']

    # eta from cfg; bivariate normal draw
    eta0 = cfg['init']['eta']
    eta_draws[0, 0] = eta0['mean'][0] + eta0['sd'][0] * np.random.randn(1)
    eta_draws[0, 1] = eta0['mean'][1]
    if eta0['sd'][1] > 0:
        eta_draws[0, 1] += (eta0['cor'] * eta0['sd'][1] / eta0['sd'][0] *
                            (eta_draws[0, 0] - eta0['mean'][0]))
        eta_draws[0, 1] += (np.sqrt(1. - eta0['cor'] ** 2) * eta0['sd'][1] *
                            np.random.randn(1))

    # Hyperparameters for state- and peptide-level variance distributions
    # directly from cfg
    shape_sigmasq[0], rate_sigmasq[0] = (
        cfg['init']['sigmasq_dist']['shape'],
        cfg['init']['sigmasq_dist']['rate'])
    shape_tausq[0], rate_tausq[0] = (cfg['init']['tausq_dist']['shape'],
                                     cfg['init']['tausq_dist']['rate'])

    # Placeholders for r and lmbda
    r[0], lmbda[0] = (0, 0)

    # Synchronize shared parameter values with workers
    # Still need to get r and lmbda squared-away after this
    params_shared = np.r_[shape_sigmasq[0], rate_sigmasq[0],
                          shape_tausq[0], rate_tausq[0],
                          r[0], lmbda[0],
                          eta_draws[0],
                          p_rnd_cen[0]]
    if supervised:
        params_shared = np.r_[params_shared, beta_draws[0],
                              mean_concentration_draws[0],
                              prec_concentration_draws[0]]

    for worker_id in xrange(1, n_workers + 1):
        comm.Send([np.array(0), MPI.INT], dest=worker_id, tag=TAGS['SYNC'])
    comm.Bcast(params_shared, root=MPIROOT)

    # Start initialization on workers
    for worker_id in xrange(1, n_workers + 1):
        comm.Send([np.array(0), MPI.INT], dest=worker_id, tag=TAGS['INIT'])

    # Initialize r and lmbda by averaging MAP estimators from workers
    r_lmbda_init = np.zeros(2)
    buf = np.zeros(2)
    comm.Reduce([buf, MPI.DOUBLE], [r_lmbda_init, MPI.DOUBLE],
                op=MPI.SUM, root=MPIROOT)
    r[0], lmbda[0] = r_lmbda_init / n_workers

    if supervised:
        # Initialize beta with Rao-Blackwellized update under default prior
        beta_draws[0] = updates_parallel.rgibbs_master_beta(
            comm=comm, **cfg['priors']['beta_concentration'])

    # Setup function for prior log density on eta, if requested
    try:
        prior_scale = cfg["priors"]["eta"]["prior_scale"]
        prior_center = cfg["priors"]["eta"]["prior_center"]
    except KeyError:
        prior_scale = None
        prior_center = None

    if prior_scale is not None:
        # Gelman's weakly-informative prior (2008)
        def dprior_eta(eta, prior_scale=5., prior_center=0.):
            return -np.log(1. + ((eta[1] - prior_center) / prior_scale)**2)

        prior_eta_kwargs = {'prior_scale': prior_scale,
                            'prior_center': prior_center}
    else:
        dprior_eta = None
        prior_eta_kwargs = {}

    # Initialize dictionary for acceptance statistics
    accept_stats = {'sigmasq_dist': 0,
                    'tausq_dist': 0,
                    'n_states_dist': 0,
                    'eta': 0}

    # Start iterations

    # Loop for MCMC iterations
    for t in xrange(1, n_iterations):
        # (0) Synchronize shared parameter values with workers
        params_shared = np.r_[shape_sigmasq[t - 1], rate_sigmasq[t - 1],
                              shape_tausq[t - 1], rate_tausq[t - 1],
                              r[t - 1], lmbda[t - 1],
                              eta_draws[t - 1],
                              p_rnd_cen[t - 1]]
        if supervised:
            params_shared = np.r_[params_shared, beta_draws[t - 1],
                                  mean_concentration_draws[t - 1],
                                  prec_concentration_draws[t - 1]]

        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id, tag=TAGS['SYNC'])
        comm.Bcast(params_shared, root=MPIROOT)

        # (1) Execute local update of protein-specific parameters on each worker
        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id, tag=TAGS['LOCAL'])

        # (2) Update state-level variance hyperparameters (sigmasq
        #   distribution). Distributed conditional independence-chain MH step.
        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id, tag=TAGS['SIGMA'])

        result = updates_parallel.rmh_master_variance_hyperparams(
            comm=comm, shape_prev=shape_sigmasq[t - 1],
            rate_prev=rate_sigmasq[t - 1], **cfg['priors']['sigmasq_dist'])
        (shape_sigmasq[t], rate_sigmasq[t]), accept = result
        accept_stats['sigmasq_dist'] += accept

        # (3) Update peptide-level variance hyperparameters (tausq
        #   distribution). Distributed conditional independence-chain MH step.
        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id, tag=TAGS['TAU'])

        result = updates_parallel.rmh_master_variance_hyperparams(
            comm=comm, shape_prev=shape_tausq[t - 1],
            rate_prev=rate_tausq[t - 1], **cfg['priors']['tausq_dist'])
        (shape_tausq[t], rate_tausq[t]), accept = result
        accept_stats['tausq_dist'] += accept

        # (4) Update parameter for negative-binomial n_states distribution (r
        #   and lmbda). Conditional independence-chain MH step.
        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id,
                      tag=TAGS['NSTATES'])

        params, accept = updates_parallel.rmh_master_nbinom_hyperparams(
            comm=comm, r_prev=r[t - 1], p_prev=1. - lmbda[t - 1],
            **cfg['priors']['n_states_dist'])
        r[t], lmbda[t] = params
        lmbda[t] = 1. - lmbda[t]
        accept_stats['n_states_dist'] += accept

        # (5) Update coefficients of intensity-based probabilistic censoring
        #   model (eta). Distributed conditional independence-chain MH step.
        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id, tag=TAGS['ETA'])

        eta_draws[t], accept = updates_parallel.rmh_master_glm_coef(
            comm=comm, b_prev=eta_draws[t - 1], MPIROOT=MPIROOT,
            propDf=prop_df_eta, prior_log_density=dprior_eta,
            prior_kwargs=prior_eta_kwargs)
        accept_stats['eta'] += accept

        # (6) Update random censoring probability. Distributed Gibbs step.
        for worker_id in xrange(1, n_workers + 1):
            comm.Send([np.array(t), MPI.INT], dest=worker_id,
                      tag=TAGS['PRNDCEN'])
        p_rnd_cen[t] = updates_parallel.rgibbs_master_p_rnd_cen(
            comm=comm, MPIROOT=MPIROOT,
            **cfg['priors']['p_rnd_cen'])

        if supervised:
            # (7) Update concentration-intensity relationship coefficients.
            #   Distributed Gibbs step.
            for worker_id in xrange(1, n_workers + 1):
                comm.Send([np.array(t), MPI.INT], dest=worker_id,
                          tag=TAGS['BETA'])
            beta_draws[t] = updates_parallel.rgibbs_master_beta(
                comm=comm, **cfg['priors']['beta_concentration'])

            if concentration_dist:
                # (8) Update concentration distribution hyperparameters
                for worker_id in xrange(1, n_workers + 1):
                    comm.Send([np.array(t), MPI.INT], dest=worker_id,
                              tag=TAGS['CONCENTRATION_DIST'])

                buf = updates_parallel.rgibbs_master_concentration_dist(
                    comm=comm, **cfg['priors']['prec_concentration'])
                mean_concentration_draws[t] = buf[0]
                prec_concentration_draws[t] = buf[1]

        # Verbose output
        if (cfg['settings']['verbose'] > 0 and
            t % cfg['settings']['verbose_interval'] == 0):
            print >> sys.stderr, 'Iteration %d complete' % t

    # Post-sampling processing and clean-up
#    # Save data from all workers
#    for worker_id in xrange(1, n_workers+1):
#        comm.Send([-1, MPI.INT], dest=worker_id, tag=TAGS['SAVE'])
    # Halt all workers
    for worker_id in xrange(1, n_workers + 1):
        comm.Send([np.array(-1), MPI.INT], dest=worker_id, tag=TAGS['STOP'])

    # Build dictionary of master-exclusive draws to return
    draws = {'eta': eta_draws,
             'p_rnd_cen': p_rnd_cen,
             'lmbda': lmbda,
             'r': r,
             'shape_tausq': shape_tausq,
             'rate_tausq': rate_tausq,
             'shape_sigmasq': shape_sigmasq,
             'rate_sigmasq': rate_sigmasq}
    if supervised:
        draws.update({
            'beta': beta_draws,
            'mean_concentration': mean_concentration_draws,
            'prec_concentration': prec_concentration_draws})

    return (draws, accept_stats, data['mapping_peptides'])


def worker(comm, rank, data, cfg):
    '''
    Worker-node process for parallel MCMC sampler.
    Receives parameters and commands from master node.
    Runs local updates and distributed components of shared draws.

    Parameters
    ----------
        - comm : mpi4py.MPI.COMM
            Initialized MPI communicator.
        - rank : int
            Rank (>= MPIROOT) of worker.
        - data : dictionary
            Data as output from load_data with rank > 0.
        - init : dictionary
            Initial parameter values as output from initialize.
        - cfg : dictionary
            Configuration dictionary containing priors, settings, and paths for
            analysis. Its format is specified in detail in separate
            documentation.

    Returns
    -------
        - draws : dictionary
            1- and 2-dimensional ndarrays containing the posterior samples for
            each protein- and ppeptide-specific parameter. Shared parameters are
            handled by the master process.
        - mapping_peptides : integer ndarray
            Worker-specific peptide to protein mapping provided in data.
        - proteins_worker : array_like, 1 dimension, nonnegative ints
            A 1d integer array of length n_proteins containing the indices (in
            the original dataset) of the proteins assigned to the given worker.
        - peptides_worker : array_like, 1 dimension, nonnegative ints
            A 1d integer array of length n_peptides containing the indices (in
            the original dataset) of the peptides assigned to the given worker.
    '''
    # Determine whether algorithm is running with supervision
    try:
        supervised = cfg['priors']['supervised']
    except KeyError:
        print >> sys.stderr, 'Defaulting to unsupervised algorithm'
        supervised = False

    # If supervised, determine whether to model distribution of concentrations
    # If this is False, prior on $\beta_1$ is scaled by $|\beta_1|^{n_{mis}}$.
    if supervised:
        try:
            concentration_dist = cfg['priors']['concentration_dist']
        except KeyError:
            print >> sys.stderr, 'Defaulting to flat prior on concentrations'
            concentration_dist = False

    # Get information on peptide features if they're available
    if 'peptide_features_worker' in data:
        n_peptide_features = data['peptide_features_worker'].shape[1]
    else:
        n_peptide_features = 0

    # Extract proposal DFs
    try:
        prop_df_y_mis = cfg['settings']['prop_df_y_mis']
    except KeyError:
        prop_df_y_mis = 5.0

    # Create references to relevant data entries in local namespace
    mapping_peptides = data['mapping_peptides']
    intensities_obs = data['intensities_obs']
    mapping_states_obs = data['mapping_states_obs']
    # Data specific to the semi-supervised algorithm
    if supervised:
        known_concentrations = data['known_concentrations']
        mapping_known_concentrations = data['mapping_known_concentrations']

    # Extract dimensions from input

    # Number of iterations from cfg
    n_iterations = cfg['settings']['n_iterations']

    # Number of peptides and proteins from mapping_peptides
    n_peptides = np.size(mapping_peptides)
    n_proteins = 1 + np.max(mapping_peptides)

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
    gamma_draws = np.empty((n_iterations, n_peptides))
    mu_draws = np.empty((n_iterations, n_proteins))

    # Concentrations, if supervised
    if supervised:
        concentration_draws = np.empty((n_iterations, n_proteins))

    # Number of censored states per peptide
    n_cen_states_per_peptide_draws = np.zeros((n_iterations, n_peptides),
                                              dtype=np.integer)

    # State- and peptide-level variances
    sigmasq_draws = np.empty((n_iterations, n_proteins))
    tausq_draws = np.empty((n_iterations, n_proteins))

    # Instantiate GLM family for eta step
    try:
        glm_link_name = cfg["priors"]["glm_link"].title()
    except KeyError:
        print >> sys.stderr, "GLM link not specified; defaulting to logit"
        glm_link_name = "Logit"
    glm_link = getattr(glm.links, glm_link_name)
    glm_family = glm.families.Binomial(link=glm_link)

    # Setup data structure for shared parameters/hyperparameters sync
    # Layout:
    #   - 0:2 : shape_sigmasq, rate_sigmasq
    #   - 2:4 : shape_tausq, rate_tausq
    #   - 4:6 : r, lmbda
    #   - 6:(6 + eta_size) : eta
    #   - +1   : p_rnd_cen
    # If supervised, 4 additional entries are used:
    #   - +2  : beta
    #   - +1  : mean_concentration
    #   - +1  : prec_concentration
    eta_size = 2 + n_peptide_features * 2
    params_shared = np.empty(7 + eta_size + 4 * supervised, dtype=np.double)

    # Prepare to receive tasks
    working = True
    status = MPI.Status()
    t = np.array(0)

    # Primary send-receive loop for MCMC iterations
    while working:
        # Receive iteration and task information
        comm.Recv([t, MPI.INT], source=MPIROOT, tag=MPI.ANY_TAG, status=status)
        task = status.Get_tag()

        if task == TAGS['STOP']:
            working = False
        elif task == TAGS['SYNC']:
            # Synchronize shared parameters/hyperparameters
            comm.Bcast(params_shared, root=MPIROOT)

            shape_sigmasq, rate_sigmasq = params_shared[0:2]
            shape_tausq, rate_tausq = params_shared[2:4]
            r, lmbda = params_shared[4:6]
            eta = params_shared[6:(6 + eta_size)]
            p_rnd_cen = params_shared[6 + eta_size]

            if supervised:
                beta = params_shared[(6 + eta_size + 1):(6 + eta_size + 3)]
                mean_concentration = params_shared[6 + eta_size + 3]
                prec_concentration = params_shared[6 + eta_size + 4]
        elif task == TAGS['INIT']:
            # Compute initial values for MCMC iterations

            # Protein-level means using mean observed intensity; excluding
            # missing peptides
            mu_draws[0] = (
                np.bincount(mapping_peptides, total_intensity_obs_per_peptide /
                            np.maximum(1, n_obs_states_per_peptide)) /
                n_obs_peptides_per_protein)
            mu_draws[0, n_obs_peptides_per_protein < 1] = np.nanmin(mu_draws[0])

            # Peptide-level means using mean observed intensity; imputing
            # missing peptides as protein observed means
            gamma_draws[0] = mu_draws[0, mapping_peptides]
            gamma_draws[0, peptides_obs] = (
                total_intensity_obs_per_peptide[peptides_obs] /
                n_obs_states_per_peptide[peptides_obs]
            )

            # State- and peptide-level variances via inverse-gamma draws
            sigmasq_draws[0] = 1. / np.random.gamma(shape=shape_sigmasq,
                                                    scale=1. / rate_sigmasq,
                                                    size=n_proteins)
            tausq_draws[0] = 1. / np.random.gamma(shape=shape_tausq,
                                                  scale=1. / rate_tausq,
                                                  size=n_proteins)

            # Mapping from protein to peptide conditional variances for
            # convenience
            var_peptide_conditional = sigmasq_draws[0, mapping_peptides]

            # Number of states parameters from local MAP estimator based on
            # number of observed peptides; very crude, but not altogether
            # terrible. Note that this ignores the +1 location shift in the
            # actual n_states distribution.
            kwargs = {
                'x': n_obs_states_per_peptide[n_obs_states_per_peptide > 0] - 1,
                'transform': True}
            kwargs.update(cfg['priors']['n_states_dist'])
            r, lmbda = lib.map_estimator_nbinom(**kwargs)
            lmbda = 1. - lmbda

            # Combine local estimates at master for initialization.
            # Values synchronize at first iteration during SYNC task.
            comm.Reduce([np.array([r, lmbda]), MPI.DOUBLE], None,
                        op=MPI.SUM, root=MPIROOT)

            if supervised:
                # Run Gibbs update on concentration-intensity coefficients using
                # noninformative prior.
                updates_parallel.rgibbs_worker_beta(
                    comm=comm, concentrations=known_concentrations,
                    gamma_bar=mu_draws[0, mapping_known_concentrations],
                    tausq=tausq_draws[0, mapping_known_concentrations],
                    n_peptides=n_peptides_per_protein[
                        mapping_known_concentrations], MPIROOT=MPIROOT)
        elif task == TAGS['LOCAL']:
            # (1) Draw missing data (n_cen and censored state intensities) given
            #   all other parameters. Exact draw via rejection samplers.

            # (1a) Obtain p_int_cen per peptide and approximatations of censored
            #   intensity posteriors.
            eta_0_effective = eta[0]
            eta_1_effective = eta[1]
            if n_peptide_features > 0:
                eta_0_effective += np.dot(data['peptide_features_worker'],
                                          eta[2:(2 + n_peptide_features)])
                eta_1_effective += np.dot(data['peptide_features_worker'],
                                          eta[(2 + n_peptide_features):])

            kwargs = {'eta_0': eta_0_effective,
                      'eta_1': eta_1_effective,
                      'mu': gamma_draws[t - 1],
                      'sigmasq': var_peptide_conditional,
                      'glm_link_name': glm_link_name}
            cen_dist = lib.characterize_censored_intensity_dist(**kwargs)

            # (1b) Draw number of censored states per peptide
            n_cen_states_per_peptide = lib.rncen(
                n_obs=n_obs_states_per_peptide,
                p_rnd_cen=p_rnd_cen,
                p_int_cen=cen_dist[
                    'p_int_cen'],
                lmbda=lmbda, r=r)
            n_cen_states_per_peptide_draws[t] = n_cen_states_per_peptide
            # Update state-level counts
            n_states_per_peptide = (n_obs_states_per_peptide +
                                    n_cen_states_per_peptide)
            n_states_per_protein = np.bincount(mapping_peptides,
                                               weights=n_states_per_peptide)
            n_states = np.sum(n_states_per_peptide)

            # (1c) Draw censored intensities
            kwargs['n_cen'] = n_cen_states_per_peptide
            kwargs['p_rnd_cen'] = p_rnd_cen
            kwargs['propDf'] = prop_df_y_mis
            kwargs.update(cen_dist)
            intensities_cen, mapping_states_cen, W = lib.rintensities_cen(
                **kwargs)

            # Sum observed intensities per peptide
            total_intensity_cen_per_peptide = np.bincount(
                mapping_states_cen, weights=intensities_cen,
                minlength=n_peptides)

            # Compute mean intensities per peptide
            mean_intensity_per_peptide = ((total_intensity_obs_per_peptide +
                                           total_intensity_cen_per_peptide) /
                                          n_states_per_peptide)

            # (2) Update peptide-level mean parameters (gamma). Gibbs step.
            gamma_draws[t] = updates_serial.rgibbs_gamma(
                mu=mu_draws[t - 1, mapping_peptides],
                tausq=tausq_draws[t - 1, mapping_peptides],
                sigmasq=var_peptide_conditional,
                y_bar=mean_intensity_per_peptide, n_states=n_states_per_peptide)
            mean_gamma_by_protein = np.bincount(mapping_peptides,
                                                weights=gamma_draws[t])
            mean_gamma_by_protein /= n_peptides_per_protein

            if supervised:
                # (3) Update concentrations given coefficients. Gibbs step.
                concentration_draws[t] = updates_serial.rgibbs_concentration(
                    gamma_bar=mean_gamma_by_protein, tausq=tausq_draws[t - 1],
                    n_peptides=n_peptides_per_protein, beta=beta,
                    mean_concentration=mean_concentration,
                    prec_concentration=prec_concentration)
                concentration_draws[t, mapping_known_concentrations] = \
                        known_concentrations

                mu_draws[t] = beta[0] + beta[1] * concentration_draws[t]
            else:
                # (3) Update protein-level mean parameters (mu). Gibbs step.
                mu_draws[t] = updates_serial.rgibbs_mu(
                    gamma_bar=mean_gamma_by_protein, tausq=tausq_draws[t - 1],
                    n_peptides=n_peptides_per_protein, **cfg['priors']['mu'])

            # (4) Update state-level variance parameters (sigmasq). Gibbs step.
            rss_by_state = ((intensities_obs -
                             gamma_draws[t, mapping_states_obs]) ** 2)
            rss_by_protein = np.bincount(mapping_peptides[mapping_states_obs],
                                         weights=rss_by_state,
                                         minlength=n_proteins)
            rss_by_state = ((intensities_cen -
                             gamma_draws[t, mapping_states_cen]) ** 2)
            rss_by_protein += np.bincount(mapping_peptides[mapping_states_cen],
                                          weights=rss_by_state,
                                          minlength=n_proteins)
            sigmasq_draws[t] = updates_serial.rgibbs_variances(
                rss=rss_by_protein, n=n_states_per_protein,
                prior_shape=shape_sigmasq, prior_rate=rate_sigmasq)

            # Mapping from protein to peptide conditional variances for
            # convenience
            var_peptide_conditional = sigmasq_draws[t, mapping_peptides]

            # (5) Update peptide-level variance parameters (tausq). Gibbs step.
            rss_by_peptide = (
                gamma_draws[t] - mu_draws[t, mapping_peptides]) ** 2
            rss_by_protein = np.bincount(mapping_peptides,
                                         weights=rss_by_peptide)
            tausq_draws[t] = updates_serial.rgibbs_variances(
                rss=rss_by_protein, n=n_peptides_per_protein,
                prior_shape=shape_tausq, prior_rate=rate_tausq)
        elif task == TAGS['SIGMA']:
            # Run distributed MH step for sigmasq hyperparameters
            updates_parallel.rmh_worker_variance_hyperparams(
                comm=comm, variances=sigmasq_draws[t], MPIROOT=MPIROOT)
        elif task == TAGS['TAU']:
            # Run distributed MH step for sigmasq hyperparameters
            updates_parallel.rmh_worker_variance_hyperparams(
                comm=comm, variances=tausq_draws[t], MPIROOT=MPIROOT)
        elif task == TAGS['NSTATES']:
            # Run distributed MH step for n_states hyperparameters
            updates_parallel.rmh_worker_nbinom_hyperparams(
                comm=comm, x=n_states_per_peptide - 1, r_prev=r,
                p_prev=1. - lmbda, MPIROOT=MPIROOT,
                **cfg['priors']['n_states_dist'])
        elif task == TAGS['ETA']:
            # Run distributed MH step for eta (coefficients in censoring model)

            # Build design matrix and response. Only using observed and
            # intensity-censored states.
            n_at_risk = n_obs_states + np.sum(W < 1)
            X = np.zeros((n_at_risk + n_peptide_features * 2,
                          2 + n_peptide_features * 2))
            X[:n_at_risk, 0] = 1.
            X[:n_at_risk, 1] = np.r_[intensities_obs, intensities_cen[W < 1]]
            if n_peptide_features > 0:
                peptide_features_by_state = data['peptide_features_worker'][
                    np.r_[mapping_states_obs, mapping_states_cen[W < 1]]
                ]
                X[:n_at_risk, 2:(2 + n_peptide_features)] = \
                    peptide_features_by_state
                X[:n_at_risk, (2 + n_peptide_features):] = \
                    (peptide_features_by_state.T * X[:n_at_risk, 1]).T
                X[n_at_risk:, 2:] = np.eye(n_peptide_features * 2)

            y = np.zeros(n_at_risk + n_peptide_features * 2)
            y[:n_obs_states] = 1.
            if n_peptide_features > 0:
                y[n_at_risk:] = 0.5

            w = np.ones_like(y)
            if n_peptide_features > 0:
                w[n_at_risk:(n_at_risk + n_peptide_features)] = (
                    cfg['priors']['eta_features']['primary_pseudoobs'] /
                    (comm.Get_size() - 1.))
                w[(n_at_risk + n_peptide_features):] = (
                    cfg['priors']['eta_features']['interaction_pseudoobs'] /
                    (comm.Get_size() - 1.))

            # Estimate GLM parameters.
            fit_eta = glm.glm(y=y, X=X, w=w, family=glm_family, info=True,
                              cov=True)

            # Handle distributed computation draw
            updates_parallel.rmh_worker_glm_coef(
                comm=comm, b_prev=eta, family=glm_family, y=y, X=X, w=w,
                MPIROOT=MPIROOT, **fit_eta)
        elif task == TAGS['PRNDCEN']:
            # Run distributed Gibbs step for p_rnd_cen
            updates_parallel.rgibbs_worker_p_rnd_cen(
                comm=comm, n_rnd_cen=np.sum(W, dtype=np.int), n_states=n_states,
                MPIROOT=MPIROOT)
        elif task == TAGS['BETA']:
            # Run distributed Gibbs step for coefficients of
            # concentration-intensity relationship
            if concentration_dist:
                updates_parallel.rgibbs_worker_beta(
                    comm=comm, concentrations=concentration_draws[t],
                    gamma_bar=mean_gamma_by_protein,
                    tausq=tausq_draws[t],
                    n_peptides=n_peptides_per_protein, MPIROOT=MPIROOT)
            else:
                updates_parallel.rgibbs_worker_beta(
                    comm=comm, concentrations=known_concentrations,
                    gamma_bar=mean_gamma_by_protein[
                        mapping_known_concentrations],
                    tausq=tausq_draws[t, mapping_known_concentrations],
                    n_peptides=n_peptides_per_protein[
                        mapping_known_concentrations], MPIROOT=MPIROOT)
        elif task == TAGS['CONCENTRATION_DIST']:
            # Run distributed Gibbs step for hyperparameters of concentration
            # distribution
            updates_parallel.rgibbs_worker_concentration_dist(
                comm=comm, concentrations=concentration_draws[t],
                MPIROOT=MPIROOT)
        elif task == TAGS['SAVE']:
            # Construct path for worker-specific results
            path_worker = cfg['output']['pattern_results_worker'] % rank

            # Setup draws to return
            draws = {'mu': mu_draws,
                     'gamma': gamma_draws,
                     'sigmasq': sigmasq_draws,
                     'tausq': tausq_draws,
                     'n_cen_states_per_peptide': n_cen_states_per_peptide_draws,
                    }
            if supervised:
                draws.update({'concentration': concentration_draws})
            lib.write_to_hdf5(
                path=path_worker, compress=cfg['output']['compress'],
                draws=draws, mapping_peptides=data['mapping_peptides'],
                proteins_worker=data['proteins_worker'])

    # Setup draws to return
    draws = {'mu': mu_draws,
             'gamma': gamma_draws,
             'sigmasq': sigmasq_draws,
             'tausq': tausq_draws,
             'n_cen_states_per_peptide': n_cen_states_per_peptide_draws,
            }
    if supervised:
        draws.update({
            'concentration': concentration_draws})

    return (draws, data['mapping_peptides'],
            data['proteins_worker'], data['peptides_worker'])


def run(cfg, comm=None):
    '''
    Coordinate parallel MCMC and output based upon process rank.

    Parameters
    ----------
        - cfg : dictionary
            Configuration dictionary containing priors, settings, and paths for
            analysis. Its format is specified in detail in separate
            documentation.
        - comm : mpi4py.MPI.COMM
            Initialized MPI communicator. If None, it will be set to
            MPI.COMM_WORLD.

    '''
    if comm is None:
        # Start MPI communications if no comm provided
        comm = MPI.COMM_WORLD

    # Get process information
    rank = comm.Get_rank()
    n_proc = comm.Get_size()

    # Load data
    data = load_data(cfg=cfg, n_workers=n_proc - 1, rank=rank)

    if rank == MPIROOT:
        # Run estimation
        draws, accept_stats, mapping_peptides = master(comm=comm,
                                                       data=data, cfg=cfg)

        # Construct path for master results
        path_master = cfg['output']['path_results_master']

        # Write master results to compressed file
        lib.write_to_hdf5(fname=path_master, compress=cfg['output']['compress'],
                          draws=draws, accept_stats=accept_stats,
                          mapping_peptides=mapping_peptides)
    else:
        result_worker = worker(comm=comm, rank=rank, data=data, cfg=cfg)
        draws, mapping_peptides = result_worker[:2]
        proteins_worker, peptides_worker = result_worker[2:]

        # Construct path for worker-specific results
        path_worker = cfg['output']['pattern_results_worker'] % rank

        # Write worker-specific results to compressed file
        lib.write_to_hdf5(fname=path_worker, compress=cfg['output']['compress'],
                          draws=draws, mapping_peptides=mapping_peptides,
                          proteins_worker=proteins_worker,
                          peptides_worker=peptides_worker)


def combine_results(result_master, list_results_workers, cfg):
    '''
    Combine MCMC results from master and workers for subsequent analysis.

    Parameters
    ----------
        - result_master : dictionary
            Dictionary containing results output by master(). This should
            contain draws, accept_stats, and mapping_peptides in their
            appropriate keys.
        - list_results_workers : list of dictionaries
            List containing dictionaries of results output by worker(). Each
            dictionary should contain draws, mapping_peptides, proteins_worker,
            and peptides_worker in their appropriate keys.
            len(list_results_workers) should equal n_workers.
        - cfg : dictionary
            Configuration dictionary used for MCMC sampling.

    Returns
    -------
        - draws : dictionary
            1- and 2-dimensional ndarrays containing the posterior samples for
            each parameter. Includes shared, protein-specific, and
            peptide-specific parameters.
        - accept_stats : dictionary
            Dictionary containing number of acceptances for each MH step.
        - mapping_peptides : integer ndarray
            Peptide to protein mapping provided in data. This is useful for
            merging worker-level results.

    '''
    # Determine whether algorithm is running with supervision
    try:
        supervised = cfg['priors']['supervised']
    except KeyError:
        print >> sys.stderr, 'Defaulting to unsupervised algorithm'
        supervised = False

    # Reference master-specific output in local scope
    draws_master = result_master['draws']
    accept_stats = result_master['accept_stats']
    mapping_peptides = result_master['mapping_peptides'][...]

    # Extract dimensions
    n_iterations = cfg['settings']['n_iterations']
    n_proteins = np.max(mapping_peptides) + 1
    n_peptides = np.size(mapping_peptides)

    # Construct combined arrays of protein-specific draws (mu, sigmasq, &
    # tausq)
    mu = np.empty((n_iterations, n_proteins))
    sigmasq = np.empty((n_iterations, n_proteins))
    tausq = np.empty((n_iterations, n_proteins))

    for result_worker in list_results_workers:
        proteins_worker = result_worker['proteins_worker'][...]

        mu[:, proteins_worker] = result_worker['draws']['mu']
        sigmasq[:, proteins_worker] = result_worker['draws']['sigmasq']
        tausq[:, proteins_worker] = result_worker['draws']['tausq']

    # Handle concentrations if running semi-supervised
    if supervised:
        concentration = np.empty((n_iterations, n_proteins))

        for result_worker in list_results_workers:
            proteins_worker = result_worker['proteins_worker'][...]
            concentration[:, proteins_worker] = \
                    result_worker['draws']['concentration']

    # Construct combined arrays of peptide-specific draws (gamma &
    # n_cen_states_per_peptide)
    gamma = np.empty((n_iterations, n_peptides))
    n_cen_states_per_peptide = np.empty((n_iterations, n_peptides))

    for result_worker in list_results_workers:
        peptides_worker = result_worker['peptides_worker'][...]

        gamma[:, peptides_worker] = result_worker['draws']['gamma']
        n_cen_states_per_peptide[:, peptides_worker] = result_worker['draws'][
                                                     'n_cen_states_per_peptide']

    # Construct dictionary of combined draws
    draws = {}

    # Master-specific draws
    draws.update(draws_master)

    # Protein-specific draws
    draws['mu'] = mu
    draws['sigmasq'] = sigmasq
    draws['tausq'] = tausq

    # Concentrations, if running semi-supervised
    if supervised:
        draws['concentration'] = concentration

    # Peptide-specific draws
    draws['gamma'] = gamma
    draws['n_cen_states_per_peptide'] = n_cen_states_per_peptide

    return draws, accept_stats, mapping_peptides


