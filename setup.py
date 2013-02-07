from distutils.core import setup

# Keeping all Python code for package in lib directory
NAME = 'quantitation'
VERSION = '0.1'
AUTHOR = 'Alexander W Blocker'
AUTHOR_EMAIL = 'ablocker@gmail.com'
URL = 'http://www.awblocker.com'
DESCRIPTION = 'Absolute quantitation for LC/MSMS proteomics via MCMC'

REQUIRES = ['numpy(>=1.6)','scipy(>=0.9)', 'yaml', 'mpi4py', 'glm']

PACKAGE_DIR = {'': 'lib'}
PACKAGES = ['quantitation','quantitation.glm']
SCRIPTS = ('mcmc_serial', 'mcmc_parallel', 'combine_results', 'summarize',
           'format_summaries')
SCRIPTS = ['scripts/quantitation_' + script for script in SCRIPTS]

setup(name=NAME,
      url=URL,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      packages=PACKAGES,
      package_dir=PACKAGE_DIR,
      scripts=SCRIPTS,
      requires=REQUIRES
      )

