
#===============================================================================
# Specific estimands
#===============================================================================

def proportion_of_concentration(mu_draws, base=10):
    '''
    Computes proportion of concentration estimand for draws, defined as:
        base**mu_i / sum(base**mu)
    for each draw of mu.

    Parameters
    ----------
    mu_draws : ndarray
        Array of posterior draws of mu, one draw per row.
    base : number
        Base of logarithm used to mu. Defaults to 10.

    Returns
    -------
    prop : ndarray
        Array of same size as mu containing the estimand.
    '''
    prop = base**mu_draws
    prop = (prop.T / prop.mean(1)).T
    return prop

