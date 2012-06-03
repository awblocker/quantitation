from distutils.core import setup

# Keeping all Python code for package in src directory

setup(name='quantitation',
      url='http://www.awblocker.com',
      version='0.1',
      description='Absolute quantitation for LC/MSMS proteomics via MCMC',
      author='Alexander W Blocker',
      author_email='ablocker@gmail.com',
      packages=['quantitation','quantitation.glm'],
      package_dir = {'': 'lib'},
      requires=['numpy(>=1.6)','scipy(>=0.9)']
      )
