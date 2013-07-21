from distutils.core import setup, Extension

import numpy

# Setup extensions
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

fast_agg_module = Extension('_fast_agg',
                            sources=['src/fast_agg_wrap.cxx',
                                     'src/fast_agg.cc'],
                            include_dirs=[numpy_include]
                           )

EXT_MODULES = [fast_agg_module]
PY_MODULES = ["fast_agg"]

# Keeping all Python code for package in lib directory
NAME = 'quantitation'
VERSION = '0.1'
AUTHOR = 'Alexander W Blocker'
AUTHOR_EMAIL = 'ablocker@gmail.com'
URL = 'http://www.awblocker.com'
DESCRIPTION = 'Absolute quantitation for LC/MSMS proteomics via MCMC'

REQUIRES = ['numpy(>=1.6)','scipy(>=0.9)', 'yaml', 'mpi4py', 'glm']

PACKAGE_DIR = {'': 'lib'}
PACKAGES = ['quantitation']
SCRIPTS = ('mcmc_serial', 'mcmc_parallel', 'mcmc_distributed',
           'combine_results', 'summarize', 'format_summaries',
           'calibrate')
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
      requires=REQUIRES,
      ext_modules=EXT_MODULES,
      py_modules=PY_MODULES
      )

