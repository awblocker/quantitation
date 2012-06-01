import numpy as np
from scipy import stats

EPS = np.spacing(1)

#==============================================================================
# Skeleton parent class 
#==============================================================================

class Link:
    '''
    Class for link functions.
    
    Encapsulates link, its inverse, and its derivative.
    
    This is a parent for all such classes
    '''
    def __call__(self, mu):
        '''
        Evaluates link function at mean mu.
        Vectorized.
        
        Placeholder.
        '''
        return NotImplementedError
    
    def inv(self, eta):
        '''
        Evaluates inverse of link function at linear predictor eta.
        Vectorized.
        
        Placeholder
        '''
        return NotImplementedError
    
    def deriv(self, mu):
        '''
        Evaluates derivative of link function with respect to mean mu.
        Vectorized.
        
        Placeholder
        '''
        return NotImplementedError

#==============================================================================
# Particular link functions
#==============================================================================

class Log(Link):
    '''
    Log link, base e.
    '''
    def __call__(self, mu):
        '''
        Log link function
        '''
        return np.log(mu)
    
    def inv(self, eta):
        '''
        Inverse of log link function (exponential)
        '''
        return np.exp(eta)
    
    def deriv(self, mu):
        '''
        Derivative of log link function with respect to mu.
        '''
        return 1./mu

class Logit(Link):
    '''
    Logit link.
    '''
    def __call__(self, mu):
        '''
        Logit link function
        '''
        return np.log(mu) - np.log(1.-mu)
    
    def inv(self, eta):
        '''
        Inverse of logit link function
        '''
        return 1./(1. + np.exp(-eta))
    
    def deriv(self, mu):
        '''
        Derivative of logit link function with respect to mu.
        '''
        return 1./mu/(1.-mu)

class Probit(Link):
    '''
    Probit link.
    '''
    bound = -stats.norm.ppf(EPS)
    
    def __call__(self, mu):
        '''
        Probit link function
        '''
        return np.minimum(self.bound,
                          np.maximum(-self.bound, stats.norm.ppf(mu)))
    
    def inv(self, eta):
        '''
        Inverse of probit link function
        '''
        return stats.norm.cdf(eta)
    
    def deriv(self, mu):
        '''
        Derivative of probit link function with respect to mu.
        '''
        return 1./np.maximum(stats.norm.pdf(stats.norm.ppf(mu)), EPS)

class Cloglog(Link):
    '''
    Cloglog link.
    '''
    def __call__(self, mu):
        '''
        Cloglog link function
        '''
        return np.log(-np.log(1.-mu))
    
    def inv(self, eta):
        '''
        Inverse of cloglog link function
        '''
        return 1 - np.exp(-np.exp(eta))
    
    def deriv(self, mu):
        '''
        Derivative of cloglog link function with respect to mu.
        '''
        return -1./np.log(1.-mu)/(1.-mu)

