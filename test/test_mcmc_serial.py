import time

import numpy as np
import yaml

import quantitation

# Set parameters
path_cfg = 'examples/basic.yml'

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

# Extract posterior means
means = {}
for k, x in draws.iteritems():
    means[k] = np.mean(x, 0)

