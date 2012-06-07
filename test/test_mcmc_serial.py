import os
import time
import bz2
import contextlib
import cPickle

import numpy as np
import yaml

import quantitation

# Set parameters
path_cfg = 'examples/basic.yml'
save = False
path_draws = 'test/draws_mcmc_serial.pickle.bz2'
path_means = 'test/means_mcmc_serial.pickle.bz2'
path_ses = 'test/ses_mcmc_serial.pickle.bz2'
path_ess = 'test/ess_mcmc_serial.pickle.bz2'

# Define functions
def effective_sample_size(draws):
    # Compute effective sample size using crude AR(1) approximation
    if len(np.shape(draws)) < 2:
        draws = draws[:,np.newaxis]
    
    # Demean the draws
    draws = draws - draws.mean(axis=0)
    
    # Compute autocorrelation by column
    acf = np.mean(draws[1:]*draws[:-1], axis=0) / np.var(draws, axis=0)

    # Compute ess from ACF
    ess = np.shape(draws)[0]*(1.-acf)/(1.+acf)
    return ess

def get_peak_mem():
    pid = os.getpid()
    status_file = open("/proc/%d/status" % pid, 'rb')
    v = status_file.read()
    status_file.close()
    i = v.index("VmPeak:")
    v = v[i:].split(None,3)
    if len(v) < 3:
        return '0'
    else: 
        return v[1]

def get_current_mem():
    pid = os.getpid()
    status_file = open("/proc/%d/status" % pid, 'rb')
    v = status_file.read()
    status_file.close()
    i = v.index("VmSize:")
    v = v[i:].split(None,3)
    if len(v) < 3:
        return '0'
    else: 
        return v[1]

print 'Starting memory:', get_current_mem(), get_peak_mem()

# Load config
cfg = yaml.load(open(path_cfg, 'rb'))

# Load data
mapping_peptides = np.loadtxt(cfg['data']['path_mapping_peptides'],
                              dtype=np.int)
mapping_states_obs, intensities_obs = np.loadtxt(cfg['data']['path_data_state'],
                                                 dtype=[('peptide', np.int),
                                                        ('intensity',
                                                         np.float)],
                                                 unpack=True)

print 'Memory after loading data:', get_current_mem(), get_peak_mem()

# Run MCMC sampler
time_start = time.time()
draws, accept_stats = quantitation.mcmc_serial(intensities_obs,
                                               mapping_states_obs,
                                               mapping_peptides, cfg)
time_done = time.time()

# Print timing information
print "%f seconds for %d iterations" % (time_done-time_start,
                                        cfg['settings']['n_iterations'])
print "%f seconds per iteration" % ((time_done-time_start) /
                                    (0.+cfg['settings']['n_iterations']))

print 'Memory after running sampler:', get_current_mem(), get_peak_mem()

# Extract posterior means
means = {}
for k, x in draws.iteritems():
    means[k] = np.mean(x, axis=0)

# Extract estimates of effective sample size
ess = {}
for k, x in draws.iteritems():
    ess[k] = effective_sample_size(x)

# Extract estimates of posterior SDs
ses = {}
for k, x in draws.iteritems():
    ses[k] = np.std(x, axis=0)

print 'Memory after computing summaries:', get_current_mem(), get_peak_mem()

if save:
    # Pickle draws and posterior summaries to compressed files

    # Draws first
    with contextlib.closing(bz2.BZ2File(path_draws, mode='wb')) as f:
        cPickle.dump(draws, file=f)

    # Means
    with contextlib.closing(bz2.BZ2File(path_means, mode='wb')) as f:
        cPickle.dump(means, file=f)

    # Standard errors
    with contextlib.closing(bz2.BZ2File(path_ses, mode='wb')) as f:
        cPickle.dump(ses, file=f)

    # Effective sample sizes
    with contextlib.closing(bz2.BZ2File(path_ess, mode='wb')) as f:
        cPickle.dump(ess, file=f)

