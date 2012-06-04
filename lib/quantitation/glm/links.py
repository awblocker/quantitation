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
    
    def deriv(self, eta):
        '''
        Evaluates derivative of mean mu with respect to linear predictor eta.
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
    
    def deriv(self, eta):
        '''
        Derivative of log link function with respect to eta.
        '''
        return np.exp(eta)

class Logit(Link):
    '''
    Logit link.
    '''

    def __call__(self, mu):
        '''
        Logit link function
        '''
        return np.log(mu/(1.-mu))
    
    def inv(self, eta):
        '''
        Inverse of logit link function
        '''
        return np.minimum(1./EPS, np.maximum(1./(1. + np.exp(-eta)), EPS))
    
    def deriv(self, eta):
        '''
        Derivative of logit link function with respect to eta.
        '''
        return np.maximum(np.exp(-eta) / (1. + np.exp(-eta))**2, np.sqrt(EPS))
        
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
    
    def deriv(self, eta):
        '''
        Derivative of normal CDF inverse link function with respect to eta.
        '''
        return np.maximum(stats.norm.pdf(eta), EPS)

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
    
    def deriv(self, eta):
        '''
        Derivative of cloglog inverse link function with respect to eta.
        '''
        eta = np.minimum(eta, 700)
        return np.maximum(np.exp(eta)*np.exp(-np.exp(eta)), EPS)

