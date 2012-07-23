import sys

import numpy as np
from mpi4py import MPI

import lib
import glm

# Set constants

# MPI constants
MPIROOT     = 0

# Tags for worker states
TAGS    = ['STOP',      # Stop iterations
           'INIT',      # Initialize before first iteration
           'SYNC',      # Synchronize hyperparameter values
           'LOCAL',     # Run local, protein-specific draws
           'PRNDCEN',   # Run distributed Gibbs update on p_rnd_cen
           'TAU',       # Run distributed MH update on tausq hyperparams
           'SIGMA',     # Run distributed MH update on sigmasq hyperparams
           'NSTATES',   # Run distributed MH update on n_states hyperparams
           'ETA',       # Run distributed MH update on eta (censoring coef)
           'SAVE'       # Save results
           ]
TAGS    = dict([(state, tag) for tag, state in enumerate(TAGS)])

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

    if rank == 0:
        # Return just the peptide to protein mapping if this is the root
        data = {'intensities_obs' : None,
                'mapping_states_obs' : None,
                'mapping_peptides' : mapping_peptides}
        return data

    # For everyone else, load the state-level data
    path_data_state = cfg['data']['path_data_state']
    mapping_states_obs, intensities_obs = np.loadtxt(path_data_state,
                                                     dtype=[('peptide', np.int),
                                                            ('intensity',
                                                             np.float)],
                                                     unpack=True)

    # Check for validity of state to peptide mapping
    if (not issubclass(mapping_states_obs.dtype.type, np.integer) or
        np.min(mapping_states_obs) < 0 or
        np.max(mapping_states_obs) > n_peptides - 1):
        raise ValueError('State to peptide mapping (mapping_states_obs)'
                         ' is not valid')

    if rank is None:
        # Return everything
        data = {'intensities_obs' : intensities_obs,
                'mapping_states_obs' : mapping_states_obs,
                'mapping_peptides' : mapping_peptides}
        return data

    # Compute observed mean intensity by protein
    n_proteins = np.max(mapping_peptides)+1
    n_obs_states_per_protein = np.bincount(mapping_peptides[mapping_states_obs],
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
    strata[strata<0] = 0
    strata[strata==n_strata] = n_strata - 1

    # Compute balanced sample of strata, caching and restoring the RNG state
    rng_state = np.random.get_state()
    np.random.seed(cfg['settings']['seed_load_data'])

    workers_by_protein = np.empty(n_proteins, dtype=np.int64)
    for stratum in xrange(n_strata):
        workers = lib.balanced_sample(np.sum(strata==stratum), n_workers)
        workers_by_protein[strata==stratum] = workers

    np.random.set_state(rng_state)

    # Subset data for individual worker
    workers_by_peptide = workers_by_protein[mapping_peptides]
    workers_by_state = workers_by_peptide[mapping_states_obs]

    # Subset intensities
    intensities_obs_worker = intensities_obs[workers_by_state==(rank-1)]

    # Subset and reindex peptide to protein mapping
    select_peptides = np.where(workers_by_peptide==(rank-1))[0]
    unique_proteins, mapping_peptides_worker = np.unique(
                                              mapping_peptides[select_peptides],
                                              return_index=False,
                                              return_inverse=True)

    # Subset and reindex state to peptide mapping
    mask_states = (workers_by_state==(rank-1))
    mapping_states_obs_worker = np.empty(np.sum(mask_states),
                                         dtype=np.int)
    reindex = dict(zip(select_peptides, range(select_peptides.size)))
    for peptide_original, peptide_worker in reindex.iteritems():
        mask_edit = (mapping_states_obs[mask_states]==peptide_original)
        mapping_states_obs_worker[mask_edit] = peptide_worker

    # Build dictionary of data to return
    data = {'intensities_obs' : intensities_obs_worker,
            'mapping_states_obs' : mapping_states_obs_worker,
            'mapping_peptides' : mapping_peptides_worker,
            'proteins_worker' : unique_proteins,
            'peptides_worker' : select_peptides}
    return data


def master(comm, data, cfg):
    '''
    Master node process for parallel MCMC. Coordinates draws, handles all
    region-level parameter draws, and collects results.

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

    # Extract number of iterations from cfg
    n_iterations = cfg['settings']['n_iterations']

    # Allocate data structures for draws

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

    # Hyperparameters for state- and peptide-level variance distributions
    # directly from cfg
    shape_sigmasq[0], rate_sigmasq[0]   = (cfg['init']['sigmasq_dist']['shape'],
                                           cfg['init']['sigmasq_dist']['rate'])
    shape_tausq[0], rate_tausq[0]       = (cfg['init']['tausq_dist']['shape'],
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

    for worker in xrange(1, n_workers+1):
        comm.Send([np.array(0), MPI.INT], dest=worker, tag=TAGS['SYNC'])
    comm.Bcast(params_shared, root=MPIROOT)

    # Start initialization on workers
    for worker in xrange(1, n_workers+1):
        comm.Send([np.array(0), MPI.INT], dest=worker, tag=TAGS['INIT'])

    # Initialize r and lmbda by averaging MAP estimators from workers
    r_lmbda_init = np.zeros(2)
    buf = np.zeros(2)
    comm.Reduce([buf, MPI.DOUBLE], [r_lmbda_init, MPI.DOUBLE],
                op=MPI.SUM, root=MPIROOT)
    r[0], lmbda[0] = r_lmbda_init / n_workers

    # Initialize dictionary for acceptance statistics
    accept_stats = {'sigmasq_dist' : 0,
                    'tausq_dist' : 0,
                    'n_states_dist' : 0,
                    'eta' : 0}

    # Start iterations

    # Loop for MCMC iterations
    for t in xrange(1, n_iterations):
        # (0) Synchronize shared parameter values with workers
        params_shared = np.r_[shape_sigmasq[t-1], rate_sigmasq[t-1],
                              shape_tausq[t-1], rate_tausq[t-1],
                              r[t-1], lmbda[t-1],
                              eta_draws[t-1],
                              p_rnd_cen[t-1]]

        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['SYNC'])
        comm.Bcast(params_shared, root=MPIROOT)


        # (1) Execute local update of protein-specific parameters on each worker
        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['LOCAL'])


        # (2) Update state-level variance hyperparameters (sigmasq
        #   distribution). Distributed conditional independence-chain MH step.
        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['SIGMA'])

        result = lib.rmh_master_variance_hyperparams(comm=comm,
                                                shape_prev=shape_sigmasq[t-1],
                                                rate_prev=rate_sigmasq[t-1],
                                                **cfg['priors']['sigmasq_dist'])
        (shape_sigmasq[t], rate_sigmasq[t]), accept = result
        accept_stats['sigmasq_dist'] += accept


        # (3) Update peptide-level variance hyperparameters (tausq
        #   distribution). Distributed conditional independence-chain MH step.
        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['TAU'])

        result = lib.rmh_master_variance_hyperparams(comm=comm,
                                                  shape_prev=shape_tausq[t-1],
                                                  rate_prev=rate_tausq[t-1],
                                                  **cfg['priors']['tausq_dist'])
        (shape_tausq[t], rate_tausq[t]), accept = result
        accept_stats['tausq_dist'] += accept


        # (4) Update parameter for negative-binomial n_states distribution (r
        #   and lmbda). Conditional independence-chain MH step.
        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['NSTATES'])

        result = lib.rmh_master_nbinom_hyperparams(comm=comm,
                                               r_prev=r[t-1], p_prev=lmbda[t-1],
                                               **cfg['priors']['n_states_dist'])
        (r[t], lmbda[t]), accept = result
        accept_stats['n_states_dist'] += accept


        # (5) Update coefficients of intensity-based probabilistic censoring
        #   model (eta). Distributed conditional independence-chain MH step.
        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['ETA'])

        eta_draws[t], accept = lib.rmh_master_glm_coef(comm=comm,
                                                       b_prev=eta_draws[t-1],
                                                       MPIROOT=MPIROOT)
        accept_stats['eta'] += accept


        # (6) Update random censoring probability. Distributed Gibbs step.
        for worker in xrange(1, n_workers+1):
            comm.Send([np.array(t), MPI.INT], dest=worker, tag=TAGS['PRNDCEN'])
        p_rnd_cen[t] = lib.rgibbs_master_p_rnd_cen(comm=comm, MPIROOT=MPIROOT,
                                                   **cfg['priors']['p_rnd_cen'])


        # Verbose output
        if (cfg['settings']['verbose'] > 0 and
            t % cfg['settings']['verbose_interval']==0):
            print >> sys.stderr, 'Iteration %d complete' % t



    # Post-sampling processing and clean-up

#    # Save data from all workers
#    for worker in xrange(1, n_workers+1):
#        comm.Send([-1, MPI.INT], dest=worker, tag=TAGS['SAVE'])

    # Halt all workers
    for worker in xrange(1, n_workers+1):
        comm.Send([np.array(-1), MPI.INT], dest=worker, tag=TAGS['STOP'])

    # Build dictionary of master-exclusive draws to return
    draws = {'eta' : eta_draws,
             'p_rnd_cen' : p_rnd_cen,
             'lmbda' : lmbda,
             'r' : r,
             'shape_tausq' : shape_tausq,
             'rate_tausq' : rate_tausq,
             'shape_sigmasq' : shape_sigmasq,
             'rate_sigmasq' : rate_sigmasq}
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
    # Create references to relevant data entries in local namespace
    mapping_peptides    = data['mapping_peptides']
    intensities_obs     = data['intensities_obs']
    mapping_states_obs  = data['mapping_states_obs']

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
    gamma_draws     = np.empty((n_iterations, n_peptides))
    mu_draws        = np.empty((n_iterations, n_proteins))

    # Number of censored states per peptide
    n_cen_states_per_peptide_draws = np.zeros((n_iterations, n_peptides),
                                              dtype=np.integer)

    # State- and peptide-level variances
    sigmasq_draws   = np.empty((n_iterations, n_proteins))
    tausq_draws     = np.empty((n_iterations, n_proteins))

    # Instantiate GLM family for eta step
    logit_family = glm.families.Binomial(link=glm.links.Logit)

    # Setup data structure for shared parameters/hyperparameters sync
    # Layout:
    #   - 0:2 : shape_sigmasq, rate_sigmasq
    #   - 2:4 : shape_tausq, rate_tausq
    #   - 4:6 : r, lmbda
    #   - 6:8 : eta
    #   - 8   : p_rnd_cen
    params_shared = np.empty(9, dtype=np.double)

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
            shape_tausq, rate_tausq     = params_shared[2:4]
            r, lmbda                    = params_shared[4:6]
            eta                         = params_shared[6:8]
            p_rnd_cen                   = params_shared[8]
        elif task == TAGS['INIT']:
            # Compute initial values for MCMC iterations

            # Protein-level means using mean observed intensity; excluding
            # missing peptides
            mu_draws[0] = (np.bincount(mapping_peptides,
                                       total_intensity_obs_per_peptide /
                                       np.maximum(1,n_obs_states_per_peptide)) /
                                       n_obs_peptides_per_protein)

            # Peptide-level means using mean observed intensity; imputing
            # missing peptides as protein observed means
            gamma_draws[0] = mu_draws[0][mapping_peptides]
            gamma_draws[0][peptides_obs] = (
                                total_intensity_obs_per_peptide[peptides_obs] /
                                n_obs_states_per_peptide[peptides_obs]
                                )

            # State- and peptide-level variances via inverse-gamma draws
            sigmasq_draws[0]    = 1./np.random.gamma(shape=shape_sigmasq,
                                                     scale=1./rate_sigmasq,
                                                     size=n_proteins)
            tausq_draws[0]      = 1./np.random.gamma(shape=shape_tausq,
                                                     scale=1./rate_tausq,
                                                     size=n_proteins)

            # Mapping from protein to peptide conditional variances for
            # convenience
            var_peptide_conditional = sigmasq_draws[0][mapping_peptides]

            # Number of states parameters from local MAP estimator based on
            # number of observed peptides; very crude, but not altogether
            # terrible. Note that this ignores the +1 location shift in the
            # actual n_states distribution.
            kwargs = {'x' : n_obs_states_per_peptide[n_obs_states_per_peptide>0]
                            - 1,
                      'transform' : True}
            kwargs.update(cfg['priors']['n_states_dist'])
            r, lmbda = lib.map_estimator_nbinom(**kwargs)

            # Combine local estimates at master for initialization.
            # Values synchronize at first iteration during SYNC task.
            comm.Reduce([np.array([r, lmbda]), MPI.DOUBLE], None,
                         op=MPI.SUM, root=MPIROOT)
        elif task == TAGS['LOCAL']:
            # (1) Draw missing data (n_cen and censored state intensities) given
            #   all other parameters. Exact draw via rejection samplers.

            # (1a) Obtain p_int_cen per peptide and approximatations of censored
            #   intensity posteriors.
            kwargs = {'eta_0' : eta[0],
                      'eta_1' : eta[1],
                      'mu' : gamma_draws[t-1],
                      'sigmasq' : var_peptide_conditional}
            cen_dist = lib.characterize_censored_intensity_dist(**kwargs)

            # (1b) Draw number of censored states per peptide
            n_cen_states_per_peptide = lib.rncen(n_obs=n_obs_states_per_peptide,
                                                p_rnd_cen=p_rnd_cen,
                                                p_int_cen=cen_dist['p_int_cen'],
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
            kwargs['propDf'] = cfg['settings']['propDf']
            kwargs.update(cen_dist)
            intensities_cen, mapping_states_cen, W = lib.rintensities_cen(**kwargs)

            # Sum observed intensities per peptide
            total_intensity_cen_per_peptide = np.bincount(mapping_states_cen,
                                                      weights=intensities_cen,
                                                      minlength=n_peptides)

            # Compute mean intensities per peptide
            mean_intensity_per_peptide = ((total_intensity_obs_per_peptide +
                                           total_intensity_cen_per_peptide) /
                                           n_states_per_peptide)


            # (2) Update peptide-level mean parameters (gamma). Gibbs step.
            gamma_draws[t] = lib.rgibbs_gamma(
                                       mu=mu_draws[t-1][mapping_peptides],
                                       tausq=tausq_draws[t-1][mapping_peptides],
                                       sigmasq=var_peptide_conditional,
                                       y_bar=mean_intensity_per_peptide,
                                       n_states=n_states_per_peptide)
            mean_gamma_by_protein = np.bincount(mapping_peptides,
                                                weights=gamma_draws[t])
            mean_gamma_by_protein /= n_peptides_per_protein


            # (3) Update protein-level mean parameters (mu). Gibbs step.
            mu_draws[t] = lib.rgibbs_mu(gamma_bar=mean_gamma_by_protein,
                                        tausq=tausq_draws[t-1],
                                        n_peptides=n_peptides_per_protein,
                                        **cfg['priors']['mu'])


            # (4) Update state-level variance parameters (sigmasq). Gibbs step.
            rss_by_state = ((intensities_obs -
                             gamma_draws[t,mapping_states_obs])**2)
            rss_by_protein = np.bincount(mapping_peptides[mapping_states_obs],
                                         weights=rss_by_state,
                                         minlength=n_proteins)
            rss_by_state = ((intensities_cen -
                             gamma_draws[t,mapping_states_cen])**2)
            rss_by_protein += np.bincount(mapping_peptides[mapping_states_cen],
                                          weights=rss_by_state,
                                          minlength=n_proteins)
            sigmasq_draws[t] = lib.rgibbs_variances(rss=rss_by_protein,
                                                n=n_states_per_protein,
                                                prior_shape=shape_sigmasq,
                                                prior_rate=rate_sigmasq)
            #bad = (~np.isfinite(sigmasq_draws[t]))
            #if np.sum(bad) > 0:
            #    bad_peptides = np.where(np.in1d(mapping_peptides,
            #                                    np.where(bad)[0]))[0]
            #    print '-'*80
            #    print ('Bad protein IDs',
            #           data['proteins_worker'][np.where(bad)[0]])
            #    print 'nStates', n_states_per_protein[bad]
            #    print 'nPeptides', n_peptides_per_protein[bad]
            #    print 'RSS', rss_by_protein[bad]
            #    print 'Shape, rate', shape_sigmasq, rate_sigmasq
            #    print 'mu', mu_draws[t][bad]
            #    print 'mean gamma', mean_gamma_by_protein[bad]
            #    print ('Obs intensity per peptide',
            #           total_intensity_obs_per_peptide[bad_peptides])
            #    print ('Cen intensity per peptide',
            #           total_intensity_cen_per_peptide[bad_peptides])
            #    print '-'*80

            # Mapping from protein to peptide conditional variances for
            # convenience
            var_peptide_conditional = sigmasq_draws[t][mapping_peptides]

            # (5) Update peptide-level variance parameters (tausq). Gibbs step.
            rss_by_peptide = (gamma_draws[t] - mu_draws[t,mapping_peptides])**2
            rss_by_protein = np.bincount(mapping_peptides,
                                         weights=rss_by_peptide)
            tausq_draws[t] = lib.rgibbs_variances(rss=rss_by_protein,
                                                  n=n_peptides_per_protein,
                                                  prior_shape=shape_tausq,
                                                  prior_rate=rate_tausq)
        elif task == TAGS['SIGMA']:
            # Run distributed MH step for sigmasq hyperparameters
            lib.rmh_worker_variance_hyperparams(comm=comm,
                                                variances=sigmasq_draws[t],
                                                shape_prev=shape_sigmasq,
                                                rate_prev=rate_sigmasq,
                                                MPIROOT=MPIROOT,
                                                **cfg['priors']['sigmasq_dist'])
        elif task == TAGS['TAU']:
            # Run distributed MH step for sigmasq hyperparameters
            lib.rmh_worker_variance_hyperparams(comm=comm,
                                                variances=tausq_draws[t],
                                                shape_prev=shape_tausq,
                                                rate_prev=rate_tausq,
                                                MPIROOT=MPIROOT,
                                                **cfg['priors']['tausq_dist'])
        elif task == TAGS['NSTATES']:
            # Run distributed MH step for n_states hyperparameters
            lib.rmh_worker_nbinom_hyperparams(comm=comm,
                                              x=n_states_per_peptide-1,
                                              r_prev=r, p_prev=lmbda,
                                              MPIROOT=MPIROOT,
                                              **cfg['priors']['n_states_dist'])
        elif task == TAGS['ETA']:
            # Run distributed MH step for eta (coefficients in censoring model)

            # Build design matrix and response. Only using observed and
            # intensity-censored states.
            n_at_risk = n_obs_states + np.sum(W<1)
            X = np.empty((n_at_risk, 2))
            X[:,0] = 1.
            X[:,1] = np.r_[intensities_obs, intensities_cen[W<1]]
            #
            y = np.zeros(n_at_risk)
            y[:n_obs_states] = 1.

            # Estimate GLM parameters.
            fit_eta = glm.glm(y=y, X=X, family=logit_family, info=True)

            # Handle distributed computation draw
            lib.rmh_worker_glm_coef(comm=comm, b_prev=eta, family=logit_family,
                                    y=y, X=X, MPIROOT=MPIROOT, **fit_eta)
        elif task == TAGS['PRNDCEN']:
            # Run distributed Gibbs step for p_rnd_cen
            lib.rgibbs_worker_p_rnd_cen(comm=comm,
                                        n_rnd_cen=np.sum(W, dtype=np.int),
                                        n_states=n_states,
                                        MPIROOT=MPIROOT)
        elif task == TAGS['SAVE']:
            # Construct path for worker-specific results
            path_worker = cfg['output']['pattern_results_worker'] % rank

            # Setup draws dictionary
            draws = {'mu' : mu_draws,
                     'gamma' : gamma_draws,
                     'sigmasq' : sigmasq_draws,
                     'tausq' : tausq_draws,
                     'n_cen_states_per_peptide':n_cen_states_per_peptide_draws,
                     }

            lib.write_to_pickle(path=path_worker,
                                compress=cfg['output']['compress_pickle'],
                                draws=draws,
                                mapping_peptides=data['mapping_peptides'],
                                proteins_worker=data['proteins_worker'])

    # Setup draws to return
    draws = {'mu' : mu_draws,
             'gamma' : gamma_draws,
             'sigmasq' : sigmasq_draws,
             'tausq' : tausq_draws,
             'n_cen_states_per_peptide':n_cen_states_per_peptide_draws,
             }
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
    data = load_data(cfg=cfg, n_workers=n_proc-1, rank=rank)

    if rank == MPIROOT:
        # Run estimation
        draws, accept_stats, mapping_peptides = master(comm=comm,
                                                       data=data, cfg=cfg)

        # Construct path for master results
        path_master = cfg['output']['path_results_master']

        # Write master results to compressed file
        lib.write_to_pickle(fname=path_master,
                            compress=cfg['output']['compress_pickle'],
                            draws=draws,
                            accept_stats=accept_stats,
                            mapping_peptides=mapping_peptides)
    else:
        result_worker = worker(comm=comm, rank=rank, data=data, cfg=cfg)
        draws, mapping_peptides = result_worker[:2]
        proteins_worker, peptides_worker = result_worker[2:]

        # Construct path for worker-specific results
        path_worker = cfg['output']['pattern_results_worker'] % rank

        # Write worker-specific results to compressed file
        lib.write_to_pickle(fname=path_worker,
                            compress=cfg['output']['compress_pickle'],
                            draws=draws,
                            mapping_peptides=mapping_peptides,
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
    # Reference master-specific output in local scope
    draws_master        = result_master['draws']
    accept_stats        = result_master['accept_stats']
    mapping_peptides    = result_master['mapping_peptides']
    
    # Extract dimensions
    n_iterations    = cfg['settings']['n_iterations']
    n_proteins      = np.max(mapping_peptides)+1
    n_peptides      = np.size(mapping_peptides)
    
    # Construct combined arrays of protein-specific draws (mu, sigmasq, & tausq)
    mu          = np.empty((n_iterations, n_proteins))
    sigmasq     = np.empty((n_iterations, n_proteins))
    tausq       = np.empty((n_iterations, n_proteins))
    
    for result_worker in list_results_workers:
        proteins_worker = result_worker['proteins_worker']
        
        mu[:,proteins_worker]       = result_worker['draws']['mu']
        sigmasq[:,proteins_worker]  = result_worker['draws']['sigmasq']
        tausq[:,proteins_worker]    = result_worker['draws']['tausq']
    
    # Construct combined arrays of peptide-specific draws (gamma &
    # n_cen_states_per_peptide)
    gamma = np.empty((n_iterations, n_peptides))
    n_cen_states_per_peptide = np.empty((n_iterations, n_peptides))
    
    for result_worker in list_results_workers:
        peptides_worker = result_worker['peptides_worker']
        
        gamma[:,peptides_worker]    = result_worker['draws']['gamma']
        n_cen_states_per_peptide[:,peptides_worker] = result_worker['draws'][
                                                     'n_cen_states_per_peptide']
    
    # Construct dictionary of combined draws
    draws = {}
    
    # Master-specific draws
    draws.update(draws_master)
    
    # Protein-specific draws
    draws['mu']         = mu
    draws['sigmasq']    = sigmasq
    draws['tausq']      = tausq
    
    # Peptide-specific draws
    draws['gamma'] = gamma
    draws['n_cen_states_per_peptide'] = n_cen_states_per_peptide
    
    return draws, accept_stats, mapping_peptides
