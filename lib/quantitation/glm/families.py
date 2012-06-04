import numpy as np
from scipy import special

import links as L

EPS = np.spacing(1)

#==============================================================================
# Skeleton parent class for all families
#==============================================================================

class Family:
    '''
    Class to encapulate a family of GLMs based on a common exponential-family
    distribution.
    
    Arguments consist of a string specifying a link function and another
    specifying a variance function.
    '''
    
    links = []
    
    def __setlink(self, link):
        '''
        Helper function to set link function while ensuring validity.
        
        link must be a character vector corresponding to a key in the links
        dictionary.
        '''
        if not isinstance(link, L.Link):
            raise TypeError('Argument is not a valid link function')
        
        if hasattr(self, 'links'):
            valid = link in self.links
            if not valid:
                raise ValueError('Argument is not a valid link function '
                                 'for this family')
        
        self._link  = link
    
    def __getlink(self):
        return self._link
    
    link = property(__getlink, __setlink)
    
    def __init__(self, link, var):
        '''
        Set link and variance function for family
        '''
        self.link   = link()
        self.var    = var
    
    def mu_init(self, y):
        '''
        Compute starting value for mu in Fisher scoring algorithm.
        
        Takes untransformed values of outcome variable as input.
        '''
        return y/2. + y.mean()/2.
    
    def weights(self, mu):
        '''
        Compute weights for Fisher scoring iterations.
        
        Takes fitted mean as input.
        '''
        return self.link.deriv(self.link(mu))**2 / self.var(mu)
    
    def loglik(self, y, mu, w=1.):
        '''
        Compute log-likelihood of observations given means mu and weights
        (input, not Fisher).
        '''
        return 
    
    def deviance(self, y, mu, w=1.):
        '''
        Compute deviance of observations given means mu and weights
        (input, not Fisher).
        '''
        return NotImplementedError

#==============================================================================
# Particular familes for GLMs
#==============================================================================

class Binomial(Family):
    '''
    Family for binomial GLMs
    '''
    links = [L.Logit, L.Probit, L.Cloglog]
    
    def __init__(self, link=L.Logit):
        self.link = link()
    
    def var(self, mu):
        '''
        Binomial variance function
        '''
        return mu*(1.-mu)
    
    def mu_init(self, y):
        '''
        Specialized initialization for binomials. Using
        
        (y + 0.5) / 2
        '''
        return (y + 0.5)/2.
    
    def loglik(self, y, mu, w=1):
        '''
        Compute log-likelihood of observations given means mu and weights
        (input, not Fisher).
        '''
        return y*np.log(mu/(1-mu) + EPS) + w*np.log(1.-mu + EPS)
    
    def deviance(self, y, mu, w=1):
        '''
        Compute deviance of observations given means mu and weights
        (input, not Fisher).
        '''
        if np.max(w) == 1:
            # Handle binary case
            return -2.*np.sum(y*np.log(mu + EPS) + (1.-y)*np.log(1.-mu + EPS))
        else:
            # Binomial case with n > 1
            return 2.*np.sum(y*np.log(y/w/mu + EPS) +
                             (w-y)*y*np.log((1.-y/w)/(1.-mu) + EPS))


class Poisson(Family):
    '''
    Family for Poisson GLMs
    '''
    links = [L.Log]
    
    def __init__(self, link=L.Log):
        self.link = link()
    
    def var(self, mu):
        '''
        Poisson variance function
        '''
        return mu
        
    def loglik(self, y, mu, w=1):
        '''
        Compute log-likelihood of observations given means mu and weights
        (input, not Fisher).
        '''
        return np.sum(w*(-mu + np.log(mu)*y - special.gammaln(y + 1)))
    
    def deviance(self, y, mu, w=1):
        '''
        Compute deviance of observations given means mu and weights
        (input, not Fisher).
        '''
        y_over_mu = y / mu
        y_over_mu[y==0] = 1.
        return 2.*np.sum(w*(y*np.log(y_over_mu) - (y-mu)))
