# Functions for the individual state level model with random censoring, common
# variance and negative binomial counts for the number of states for each
# peptide.

from math import log
from math import pi
from math import exp
from math import sqrt

import numpy as np

from scipy import linalg
from scipy import stats
from scipy import optimize
from scipy import special

def logit_censored_normal_pdf(x, eta_0, eta_1, mu, sigma):
    return (
        ((2*pi*sigma**2)**(-0.5) * exp((-(x-mu)**2 / (2*sigma**2)))) *
            (1 + exp(eta_0 + eta_1*x))**(-1)
    )
    
def logit_observed_normal_pdf(x, eta_0=-3, eta_1=2, mu=-1, sigma=1):
    return (
        ((2*pi*sigma**2)**(-0.5) * exp((-(x-mu)**2 / (2*sigma**2)))) * 
            (exp(eta_0 + eta_1*x) / (1 + exp(eta_0 + eta_1*x)))
    )

def p_censored(x, eta_0, eta_1):
    return (1/(1 + exp(eta_0 + eta_1*x)))

# Derivative of the log density ratio.
def dLogTargPropRatio(x, eta_0, eta_1, mean, sd, y_hat, approx_sd, df):
    return (
        -(x - mean)/sd**2 + 
            ((1 + df)*(x - y_hat)) / (df*approx_sd**2 + (x - y_hat)**2) - 
            (eta_1 * exp(eta_0 + eta_1*x)) / (1 + exp(eta_0 + eta_1*x))
    )

# Compute and return the log likelihood.
def dCensoredIntensityLogLhood(x, args):
    # First derivativ of the log likelihood for censored intensities
    eta_0 = args['eta_0'][0]
    eta_1 = args['eta_1'][0]
    mu = args['mu'][0]
    sigma = args['sigma'][0]
    # Compute the log likelihood.
    return ((-1 + 1/(1 + exp(eta_0 + eta_1*x))) * eta_1 - (x - mu)/sigma**2)
    
# Returns incorrect p_int_censored value
def characterizeCensoredIntensityDistribution(eta_0, eta_1, mu, sigma, TOL=1e-5,
                                              MAX_ITER=200):
    # 0) Run a few iterations of bisection.
    x_0 = vectorizedBisection(f=dCensoredIntensityLogLhood,
                              f_args=np.array([(eta_0, eta_1, mu, sigma)],
                                           dtype=[('eta_0', int),
                                                  ('eta_1', int),
                                                  ('mu', int),
                                                  ('sigma', int)]),
                              lower=mu - 5*sigma, upper=mu + 5*sigma, TOL=1e-4,
                              MAX_ITER=10)
    
    # 1) Compute y_hat, the mode, using Halley's Method.
    N = 0
    STOP_ITER = False
    x_n = x_0
    while N <= MAX_ITER and STOP_ITER == False:
        N = N + 1
        x_nm1 = x_n
        # Evaluate the first, second and third derivative of the log-lik
        g_prime = float(((-1 + 1/(1 + exp(eta_0 + eta_1*x_nm1))) *
                         eta_1 -(x_nm1 - mu)/(sigma**2)))
        g_doubleprime = float((-sigma**(-2) -
                               (eta_1**2 * exp(eta_0 + eta_1* x_nm1) ) /
                               (1 + exp(eta_0 + eta_1 * x_nm1))**2))
        g_tripleprime = float(((2 * eta_1**3 *exp(2*eta_0 + 2*eta_1*x_nm1)) /
                               (1 + exp(eta_0 + eta_1*x_nm1))**3 -
                               (eta_1**3 * exp(eta_0 + eta_1*x_nm1)) /
                               (1 + exp(eta_0 + eta_1*x_nm1))**2))
        x_n = float((x_nm1 - (2.0*g_prime*g_doubleprime) /
                     (2.0*g_doubleprime**2 - g_prime*g_tripleprime)))
        
        if abs(g_prime) < TOL:
            # Save the modes.
            y_hat = x_n
            print 'Halley\'s Method Iterations:', N
            STOP_ITER = True
    
        if STOP_ITER == False:
            print 'Bad Initialization.'
            y_hat = x_n
        
    # 2) Compute approximate standard deviation.
    approx_sd = 1
    approx_sd = sqrt(-1.0/g_doubleprime)
    
    # 3) Compute approximate normalizing constant, which is P(Int. Cen.) using
    # Laplace approximation.
    p_int_censored = (sqrt(2.0*pi/(-( -sigma**-2 - (eta_1**2 * exp(eta_0 + eta_1
                                                                   * y_hat)) /
                                     (1.0 + exp(eta_0 + eta_1 * y_hat))**2))) *
                      ((1.0+exp(eta_0 + eta_1*y_hat))**(-1)) *
                      ((2.0*pi*sigma**2)**(-1/2) * exp(-(y_hat-mu)**2 /
                                                       (2*sigma**2))))
    
    return y_hat, p_int_censored, approx_sd
    
def computePCenViaMonteCarloIntegration(random_normals, mu, sigma, eta_0, eta_1):
    p_cens = 1/(1 + exp(eta_0 + eta_1*(random_normals*sigma + mu)))
    p_int_censored = np.mean(p_cens)
    return p_int_censored
    
def dLogTargPropRatio_env(x, args):
    # First derivative of the log target-proposal ratio, using an environment
    # to contain the additional arguments to the function
    eta_0 = args['eta_0'][0]
    eta_1 = args['eta_1'][0]
    mu = args['mu'][0]
    sigma = args['sigma'][0]
    approx_sd = args['approx_sd'][0]
    y_hat = args['y_hat'][0]
    df = args['df'][0]
    return (
        (-(x - mu)/(sigma**2)) +
        (((1 + df)*(x - y_hat)) / (df*approx_sd**2 + (x - y_hat)**2)) -
        (eta_1 * exp(eta_0 + eta_1*x)) / (1 + exp(eta_0 + eta_1*x))
    )
    
def vectorizedBisection(f, f_args, lower, upper, TOL=1e-10, MAX_ITER=100):
    # Find multiple roots using bisection. f should be a function that takes
    # argument x and f_args.
    
    # Check if the starting points are valid.
    same_sign = 0
    if isinstance(lower, float) == False:
        for i in range(len(lower)):
            if sum(sign(f(lower[i], f_args))) == sum(sign(f(upper[i], f_args))):
                same_sign = same_sign + 1
    if (same_sign != 0):
        print 'Invalid Starting Points.'
    
    t = 0
    
    # Initialize active vector, 1 is TRUE and 0 is FALSE, along with check vectors
    if isinstance(lower, float):
        active = error = mid = f_c = 1
        while t <= MAX_ITER and active == 1:
            t = t + 1
            # Get the midpoint and determine which values have converged
            mid = (lower + upper) / 2.0
            buffer = mid
            f_c = f(mid, f_args)
            if sign(f_c) == sign(f(lower, f_args)):
                lower = buffer
            else:
                upper = buffer
            error = abs(upper - lower) / 2.0
            if error < TOL:
                active = 0
                mid = (lower + upper) / 2.0
                
    else:
        active = error = mid = f_c = [1]*len(lower)
        while t <= MAX_ITER and max(active) > 0: 
            t = t + 1
            # Get the midpoints and determine which values have converged.
            for i in range(len(lower)):
                if active[i] == 1:
                    t = t
                    mid[i] = (lower[i] + upper[i]) / 2.0
                    # Update values.
                    buffer = mid[i]
                    f_c[i] = f(buffer, f_args)
                    if sum(sign(f_c[i])) == sum(sign(f(lower[i], f_args))):
                        lower[i] = buffer
                    else:
                        upper[i] = buffer
                    # Calculate resulting error.
                    error[i] = abs(upper[i] - lower[i]) / 2.0
                    if error[i] < TOL:
                        active[i] = 0
                        mid[i] = (lower[i] + upper[i]) / 2.0
    mid = np.array(mid)
    return mid
    
def computeM(eta_0, eta_1, mu, sigma, y_hat, approx_sd, normalizing_cnst, df, TOL=1e-10, MAX_ITER=100):
    # Compute the maximum of the target proposal density rato for the
    # accept-reject algorithm
    
    # Store arguments for the first derivative function
    d1_args = array([(eta_0, eta_1, mu, sigma, approx_sd, y_hat, df)],
                    dtype=[('eta_0', int), ('eta_1', int), ('mu', int),
                           ('sigma', int), ('approx_sd', int),
                           ('y_hat', object), ('df', int)])
    
    # Initialize vectors for all four of the bounds
    left_lower = left_upper = right_lower = right_upper = np.zeros(len(y_hat))
    
    # Make sure the starting points are the correct sign
    left_lower = y_hat - approx_sd
    left_upper = y_hat - 10*TOL
    right_lower = y_hat + 10*TOL
    right_upper = y_hat + approx_sd
    for i in range(len(y_hat)):
        while dLogTargPropRatio_env(left_lower, d1_args)[i] < 0:
            left_lower[i] = left_lower[i] - approx_sd
        while dLogTargPropRatio_env(left_upper, d1_args)[i] > 0:
            left_upper[i] = left_upper[i] - 10*TOL
        while dLogTargPropRatio_env(right_lower, d1_args)[i] < 0:
            right_lower[i] = right_lower[i] + 10*TOL
        while dLogTargPropRatio_env(right_upper, d1_args)[i] > 0:
            right_upper[i] = right_upper[i] + approx_sd

    # Find zeros that are less than y_hat using bisection.
    left_roots = vectorizedBisection(f=dLogTargPropRatio_env, lower=
                                     left_lower, upper=left_upper, f_args=
                                     d1_args, TOL=TOL, MAX_ITER=MAX_ITER)
    
    # Find zeros that are greater than y_hat using bisection.
    right_roots = vectorizedBisection(f=dLogTargPropRatio_env, lower =
                                      right_lower, upper=right_upper, f_args =
                                      d1_args, TOL=TOL, MAX_ITER=MAX_ITER)
    
    print (left_roots - y_hat)/approx_sd
    
    # Compute M.
    f_left_roots = (normalizing_cnst * logit_censored_normal_pdf(left_roots,
                                                                 eta_0, eta_1,
                                                                 mu, sigma) /
                    t.pdf((left_roots - y_hat)/approx_sd, 1))
    f_right_roots = (normalizing_cnst * logit_censored_normal_pdf(right_roots,
                                                                  eta_0, eta_1,
                                                                  mu, sigma) /
                     t.pdf((left_roots - y_hat)/approx_sd, 1))
    
    # Store maximum of each root
    M = np.zeros(len(f_left_roots))
    for i in range(len(M)):
        M[i] = max(f_left_roots[i], f_right_roots[i])
        
    # Return results
    return(M, left_roots, right_roots)

def targetProposalRatio(x, mu, sigma, y_hat, approx_sd, eta_0, eta_1, normalizing_cnst, df):
    # Ratio of censored peptide likelihood to the proposal t density.
    return (
        (normalizing_cnst * norm.pdf(x, loc=mu, scale=sigma) *
            (1 + exp(eta_0 + eta_1*x))**(-1)) /
            (t.pdf((x - y_hat)/approx_sd, df))
    )

def censoredPeptideAcceptRejectSampler(n, mu, sigma, y_hat, approx_sd, normalizing_cnst, eta_0, eta_1, M, df):
    # Accept-reject sampler for censored peptide intensities.
    samples = np.zeros(n)
    n_samples = 0
    
    # Draw Samples until the desired number of accepted draws is achieved.
    draws = 0
    while n_samples < n:
        draws = draws + 1
        proposal = t.rvs(df)*approx_sd + y_hat
        print proposal
        unif = uniform.rvs()
        accept_prob = (1/M) * targetProposalRatio(proposal, mu, sigma, y_hat,
                                                  approx_sd, normalizing_cnst,
                                                  eta_0, eta_1, df)
        print accept_prob
        # Check if proposal is accepted.
        if unif < accept_prob: 
            n_samples = n_samples + 1
            samples[n_samples-1] = proposal
        acceptance_rate = n_samples / draws
    return samples, acceptance_rate
        
def drawNCensored(p_ic, p_rc, n_obs, lmbda, r):
    # Draw the number of censored peptide states.
    # If no states were observed for the peptide, then the support of the
    # geometric is 1, 2, 3,..., for for peptides with at least one observed state
    # the support is 0, 1, 2,...
    
    n_draws_total = max(1, len(p_ic), len(n_obs))
    n_censored_draws = ones(n_draws_total)
    geom_p = 1-(1-lmbda)*(p_rc + (1-p_rc)*p_ic)
    n_obs_g_0 = n_obs > 0
    n_censored_draws = random.binomial(n_draws_total, geom_p, size=n_obs + r)
    n_censored_draws = n_censored_draws + n_obs_g_0
    return n_censored_draws
    
#def rnCen(nobs, prc, pic, lmbda, r):
    
def drawNCensoredMH(p_ic, n_obs, lmbda, n_cen_tm1):
    # Draw n censored using Metropolis-Hastings
    geom_p = lmbda
    n_draws = max(1, len(p_ic), len(n_obs))
    # Draw n cens.
    n_cen_star = random.geometric(geom_p, n_draws) + (n_obs == 0)
    # Calculate the log of the importance ratio minus the prior ratio
    LPR_m_LIR = (n_cen_star - n_cen_tm1) * log(p_ic)
    logU = log(random.uniform(size = n_draws))
    # Set n_cen(t) according to the MH update rule
    update = logU < LPR_m_LIR
    print update
    n_cen_t = n_cen_tm1
    n_cen_t[update] = n_cen_star[update]
    return(n_cen_t)

def drawCensoredIntensitiesFast(n, mu, sigma, y_hat, approx_sd, p_ic, p_rc,
                                eta_0, eta_1, df, protein, peptide):
    # Draws the intensities of censored peptides.
    censored_intensities_dtype = dtype( [('Protein', '|S3'), ('Peptide', '|S3'),
                                         ('State', '|S3'),
                                         ('Intensity', float64),
                                         ('Observed', int), ('W', int)])
    censored_intensities = np.zeros(sum(n), censored_intensities_dtype)
    # First, get the maximum of the target, proposal ratio.
    computeMResults = computeM(
        eta_0=eta_0,
        eta_1=eta_1, 
        mu=mu,
        sigma=sigma,
        y_hat=y_hat,
        approx_sd=approx_sd,
        normalizing_cnst=1/p_ic,
        df=df,
        TOL=1e-10,
        MAX_ITER=100)
    M = computeMResults['M']
    # Get the number of peptides that we need to draw intensities for.
    n_peps = max(1, len(mu), len(sigma), len(n), len(approx_sd), len(y_hat))
    n_draws_total = sum(n)
    # Create a dataframe of arguments to use in the accept-reject sampler.
    accept_reject_dtype = dtype(
        [('Intensity', float64),
         ('mu', float64),
         ('sigma', float64),
         ('y_hat', float64),
         ('approx_sd', float64),
         ('p_ic', float64),
         ('M', float64),
         ('W', int),
         ('Protein', '|S3'), 
         ('Peptide', '|S3'), 
         ('accept_probs', float64)])
    accept_reject_df = np.zeros(sum(n), accept_reject_dtype)
    for i in range(len(n_peps)):
        index = 0        
        accept_reject_dtype['mu'][index:(index+n[i]+1)] = mu[i]
        accept_reject_dtype['sigma'][index:(index+n[i]+1)] = sigma[i]
        accept_reject_dtype['y_hat'][index:(index+n[i]+1)] = y_hat[i]
        accept_reject_dtype['approx_sd'][index:(index+n[i]+1)] = approx_sd[i]
        accept_reject_dtype['normalizing_cnst'][index:(index+n[i]+1)] = normalizing_cnst[i]
        accept_reject_dtype['p_ic'][index:(index+n[i]+1)] = p_ic[i]
        accept_reject_dtype['M'][index:(index+n[i]+1)] = M[i]
        accept_reject_dtype['Protein'][index:(index+n[i]+1)] = Protein[i]
        accept_reject_dtype['Peptide'][index:(index+n[i]+1)] = Peptide[i]
    
    # Draw the random censoring indicators.
    accept_reject_df['W'] = random.binomial(1, p_rc / (p_rc + (1-p_rc)*accept_reject_df['p_ic'], n_draws_total))
    
    # Draw the randomly censored intensities.
    n_rand_censored = sum(accept_reject_df['W'])
    if  (n_rand_censored > 0):
        for i in range(n_draws_total):
            if accept_reject_df['W'][i] == 1:
                accept_reject_df['Intensity'][i] = random.normal(
                    accept_reject_df['mu'][i], 
                    accept_reject_df['sigma'][i],
                    size = 1)
            else:
                while accept_reject_df['intensity'][i] == 0:
                    unif_draw = random.uniform()
                    proposed_intensity = t.rvs(df,0,1)*accept_reject_df['approx_sd'][i] + accept_reject_df['y_hat'][i]
                    accept_probs = (1/(accept_reject_df['M'][i])) * (targetProposalRatio(
                        proposed_intensity,
                        accept_reject_df['mu'][i], 
                        accept_reject_df['sigma'][i], 
                        accept_reject_df['y_hat'][i], 
                        accept_reject_df['approx_sd'][i], 
                        eta_0, 
                        eta_1,
                        1/(accept_reject_df['p_ic'][i]), 
                        df))
                    if unif_draw < accept_probs:
                        accept_reject_df['intensity'][i] = proposed_intensity
    
    censored_intensities['Protein'] = accept_reject_df['Protein']
    censored_intensities['Peptide'] = accept_reject_df['Peptide']
    censored_intensities['State'] = 'Censored_State'
    censored_intensities['Observed'] = False
    censored_intensities['Intensity'] = accept_reject_df['Intensity']
    censored_intensities['W'] = accept_reject_df['W']
    
    return censored_intensities
    
def computeMeanAndSd(x):
    # Function to compute the mean and standard deviation of a vector.
    n = len(x)
    x_bar = np.mean(x)
    sd = sqrt(sum((x-x_bar)^2) / (n-1))
    return x_bar, sd

def profLikNbinom(r, x, r_prior=np.array([1,0])):
    # Profile likelihood for r, with lambda fixed at its MLE.
    # Note: This assumes that x = 1, 2,.... instead of the usual
    # support of x = 0, 1, 2, 3,...
    n = len(x)
    p_hat = r/(np.mean(x) + r - 1)
    return (sum(log(nbinom.pmf(x-1, r, p_hat))) + (r_prior[0]-1)*log(r) - r_prior[1]*exp(log(r)))
    
# Setup actual log-likelihood
def loglikNbinom(logitP, logR, x, r_prior = np.array([1,0])):
    # Negative binomial log likelihood.
    # Note: This again assumes that x = 1, 2,..., instead of the usual
    # support of x = 0, 1, 2, 3,...
    p = 1/(1+exp(-logitP))
    r = exp(logR)
    return (sum(log(nbinom.pmf(x-1, r, p))) + (r_prior[0]-1)*logR - r_prior[1]*exp(logR))

# Functions for computing the information matrix fro the reparametrized model
# parameters, specifically logit(p) and log(r). 
def d2P(p,r,n,x):
    return_value = ((-n*r)/p**2 - (sum(x)-n)/(1-p)**2 * (p * (1-p))**2 +
        ((n*r)/p - (sum(x) - n)/(1-p)) * (exp(log(p/(1-p))) * 
        (1-exp(log(p/(1-p)))) / (1+exp(log(p/(1-p))))**3))
    return return_value
    
def d2R(p,r,n,x, r_prior = np.array([1,0])):
    return ((sum(digamma(x+r-1)) - n * digamma(r) + n  * log(p) + 
        (r_prior[0] - 1)/r - r_prior[1]) * r + 
        (sum(polygamma(1,x+r-1)) - n * polygamma(1,r) - (r_prior[0]-1)/r**2) * r**2)
        
def crossPR(p,r,n):
    return ((p*(1-p)) * r * (n/p))

# Setup information matrix    
def infoMat(theta,x,r_prior=np.array([1,0])):
    p = exp(theta[0])/(1+exp(theta[0]))
    r = exp(theta[1])
    n = len(x)
    return (-(np.matrix[[d2P(p,r,n,x), crossPR(p,r,n)], [crossPR(p,r,n), d2R(p,r,n,x,r_prior)]]))

    
def computeNbinomMLEs(x, r_prior = np.array([0,1])):
    # Maximize the profile likelihood of r to obtain the MLEs of logit(p) and 
    # log(r). Also computes their covaraince matrix using the negative inverse
    # of the observed information matrix.
    
    # Compute the MLEs.
    rMLE = brent(-profLikNbinom, args=(x, r_prior,), brack=(0.01, 50))
    nbinomMLEs = np.array([(rMLE)/(np.mean(x) + rMLE - 1), rMLE])
    # Compute the covariance matrix
    covMat = linalg.solve_triangular(infoMat(np.array([log(nbinomMLEs[0]/(1-nbinomMLEs[0])),
        log(nbinomMLEs[1])]), x, r_prior=r_prior))
    logitLambdaMLE = log(nbinomMLEs[0]/(1-nbinomMLEs[0]))
    logRMLE = log(nbinomMLEs[1]),
    covMat = covMat
    return logitLambda, logRMLE, covMat
    
def lambdaAndRMH(n_by_pep, lambda_tm1, r_tm1, r_prior= np.array([1,0])):
    # Compute the MLE for the negative binomial parameters. Then draw proposals
    # for (logit(lambda), log(r)) using the normal approximation. Finally, perform
    # the MH step and return (lambda[t], r[t]).
    
    # Compute the MLEs for logit(p) and log(r).
    MLEs = computeNbinomMLEs(n_by_pep, r_prior = r_prior)
    logitLambdaMLE = MLEs['logitLambdaMLE']
    logRMLE = MLEs['logRMLE']
    covMat = MLEs['covMat'] * 1000
    print 'covMat:'
    print covMat
    print 'corMat:'
    print cov2cor(covMat)
    # Draw proposals for logit(lambda) and log(r)
    proposals = random.multivariate_normal(array([logitLambdaMLE, logRMLE]), covMat)
    
    print 'r MLE:', exp(logRMLE)
    print 'r proposal:', exp(proposals[1])
    print 'lambda MLE:', 1/(1+exp(-logitLambdaMLE))
    print 'lambda proposal:', 1/(1+exp(proposals[0]))
    
    # Metropolis-Hasting.
    lpr = (loglikNbinom(logP=proposals[0], logR=proposals[1], x=n_by_pep,
                        r_prior = r_prior) -
           loglikNbinom(logitP=log(lambda_tm1/(1-lambda_tm1)), logR=log(r_tm1),
                        x=n_by_pep, r_prior=r_prior))
    print 'lpr:', lpr
    prop_r = exp(proposals[1])
    prop_p = 1/(1+exp(-proposals[0]))
    
    lir = (dmvnorm(log(x), np.array([logitLambdaMLE, logRMLE]), covMat) -	
        log(prop_r) - log(prop_p) - log(1-prop_p) -
        (dmvnorm(np.array([log(lambda_tm1/(1-lambda_tm1)), log(r_mt1)]),
                 np.array([logitLambdaMLE, logRMLE]),
                 covMat) - log(r_tm1) - log(lambda_tm1) - log(1-lambda_tm1)))
                 
    print 'lir:', lir
    # MH Update.
    print 'lpr - lir:', lpr-lir
    log_u = log(random.uniform())
    if log_u < (lpr_lir):
        lambda_t = 1/(1 + exp(-proposals[0]))
        r_t = exp(proposals[1])
    else:
        lambda_t = lambda_tm1
        r_t = r_tm1
    
    return array([lambda_t, r_t])
    
# Use the profile likelihood to draw log(r), then use the conditional draw for
# lambda|log(r) to get lambda.  Then do the metropolis step jointly.

# Set up the information matrix for log(r).
def logRInformation(r, x, r_prior= np.array([1,0])):
    # Information matrix for log(r), using the negative binomial profile 
    # likelihood.
    n = len(x)
    n_bar = np.mean(x)
    d1r_profile = (sum(digamma(x + r -1)) - n*digamma(r) + 
        n*( (n_bar-1)/(n_bar+r-1) + log(r/(n_bar+r-1)) ) -
        sum(x-1)/(n_bar+r-1) + (r_prior[0]-1)/r - r_prior[1])
    d2r_profile = (sum(polygamma(1,x+r-1)) - n*polygamma(1,r) +
            n*(n_bar-1)**2/(r*(n_bar+r-1)**2) +
            sum(x-1)/(n_bar+r-1)**2 - (r_prior[0] -1)/r**2)
    iMat_profile <- (d1r_profile*r + d2r_profile*r**2)
    return(iMat_profile)
    
def rLambdaMH2(n_by_pep, lambda_tm1, r_tm1, r_prior = np.array([1,0])):
    rMLE = brent(-profLikNbinom, args=(n_by_pep, r_prior,), brack=(0.01, 50))
    logRMLE = log(rMLE)
    logRMLEVar = 1/(-logRInformation(rMLE, n_by_pep, r_prior))
    
    # Draw the proposal for log(r).
    r_star = np.random.lognormal(logRMLE, sqrt(logRMLEVar))
    
    # Draw lambda|r_star.
    n_peptides_total = len(n_by_pep)
    n_states_total = sum(n_by_pep)
    
    lambda_tar = np.random.beta(r_star*n_peptides_total + 1, n_states_total-n_peptides_total+1)
    
    # Compute the prior ratio.
    lpr = (sum(log(nbinom.pdf(n_by_pep-1, r_star, lambda_star))) + (r_prior[0]-1)*log(r_star) -
          r_prior[1]*r_star -
          (sum(log(nbinom.pdf(n_by_pep-1, r_tm1, lambda_tm1))) + (r_prior[0]-1)*log(r_tm1) -
          r_prior[1]*r_tm1))
        
    # Compute the importance ratio.
    lir = (log(lognormal_pdf(r_star, logRMLE, sqrt(logRMLEVar))) +
          log(beta.pdf(lambda_star, r_star*n_peptides_total+1, n_states_total - n_peptides_total+1)) -
          log(lognormal_pdf(r_tm1, logRMLE, sqrt(logRMLEVar))) -
          log(beta.pdf(lambda_tm1, r_tm1*n_peptides_total+1, n_shapes_total-n_peptides_total+1)))
          
    # MH Step.
    log_u = log(random.uniform())
    if log_u < (lpr - lir):
        lambda_t = lambda_star
        r_t = r_star
    else:
        lambda_t = lambda_tm1
        r_t = r_tm1
    return np.array([lambda_t, r_t])
    
    #--- Hierarchical model updates. ---#

# Gamma hyper prior update.
def profLikGamma(a, x):
    # Profile log likelihood of the shape parameter of a gamma distribution, with
    # mean shape/rate.
    b_hat = (a - 1/len(x))/np.mean(x)
    return (sum(log(gamma.pdf(x, a, scale=1/b_hat)) - log(b_hat) +
            lognormal_pdf(a, 2.65, 0.652)))
    
def logLikelihoodGamma(logA, logB, x):
    # Log Likelihood for a gamma with the shape and rate log transformed.
    return (sum(log(gamma.pdf(x, exp(logA), scale=1/exp(logB)))))
    
def infoMatGamma(theta,x):
    # Information matrix for log transformed gamma parameters.
    a = exp(theta[0])
    b = exp(theta[1])
    n = len(x)
    
    iMat = (-matrix([[a*(n*log(b)-n*digamma(a)+sum(log(x)) + ((-1*(0.652**2)-log(a)+2.65)/(0.652**2 * a))) +
        a*a*(-n*polygamma(1,a) + ((-1-2.65 + 0.652**2 + log(a))/(a**2 * (0.652)**2))), a*n], [a*n, -b*sum(x)]]))
    return iMat

def dmvnorm(x, mean, sigma, Log=True):
    z = linalg.solve_triangular(sigma, x-mean, lower=False)
    density = (-(len(x)/2)*log(2*pi)-(1/2)*(np.sum(np.log(np.diag(sigma)))) - 0.5 * np.sum(z*z))
    if Log==True:
        return density
    else:
        return exp(density)    
   
def Logmvlnorm(x, logmean, logsigma):
    return dmvnorm(log(x), logmean, logsigma) - sum(log(x)) 
    
def varianceHyperparametersMHUpdate(variances, alpha_tm1, beta_tm1):
    # Metropolis-Hastings update for the variance hyper parameters, using the
    # normal approximateion for the log of their MLEs.
    precisions = 1 / variances
    # Compute the MLE for the shape parameter using the profile likelihood.
    alphaMLE = brent(-profLikGamma, args=(precisions,), brack=(0.01, 10000))
    logAlphaMLE = log(alphaMLE)
    # Use the MLE for the shape to compute the MLE for the rate.
    betaMLE = (alphaMLE - 1/len(variances))/np.mean(precisions)
    print 'alpha mle, beta mle:', alphaMLE, betaMLE
    logBetaMLE = log(betaMLE)
    # Calculate the observed information matrix for the log transformed parameters.
    infoMat = infoMatGamma(theta=np.array([logAlphaMLE, logBetaMLE]), x=precisions)
    covMat = linalg.inv(infoMat)
    print 'Info matrix:'
    print infoMat
    print 'Correlation Matrix:'
    print cov2cor(covMat)
    # Draw proposals for alpha and beta using the log normal centered around
    # their logMLEs.
    proposals = exp(random.multivariate_normal(np.array([logAlphaMLE, logBetaMLE]), covMat))
    print 'alpha star, beta star:', proposals[0], proposals[1]
    alphaStar = proposals[0]
    betaStar = proposals[1]
    # Compute the prior ratio.
    lpr = (sum(log(gamma.pdf(precisions, alphaStar, 1/betaStar))) -
        log(betaStar) + log(lognormal_pdf(alphaStar, 2.65, 0.652)) -
        sum(log(gamma.pdf(precisions, alpha_tm1, 1/beta_tm1))) +
        log(betaStar) - log(lognormal_pdf(alpha_tm1, 2.65, 0.652)))
    print 'lpr:', lpr
    lir = (Logmvlnorm(x=proposals, logmean=np.array([logAlphaMLE, logBetaMLE]), logsigma=covMat) -
        Logmvlnorm(x=np.array([alpha_tm1, beta_tm1]), logmean=np.array([logAlphaMLE, logBetaMLE]), logsigma=covMat))
    print 'lir:', lir
    print 'lpr-lir:', lpr-lir
    # MH Step.
    log_u = log(random.uniform())
    print 'MH Update:', (log_u < (lpr-lir))
    if log_u < (lpr-lir):
        alpha_t = alphaStar
        beta_t = betaStar
    else:
        alpha_t = alpha_tm1
        beta_t = beta_tm1
    
    return np.array([alpha_t, beta_t])

def gammaGibbsUpdate(mu, tau, sigma, y_bar, n_states):
    # Determine the length of the arguments passed for vectorization.
    n_draws = max(1, len(mu), len(tau), len(sigma), len(y_bar), len(n_states))
    
    # Compute the mean and variance of the conditional distribution of gamma...
    g_mean = ((mu/tau**2) + (y_bar/(sigma**2/n_states))) / ((1/tau**2) + (n_states/sigma**2))
    g_var = 1/((1/tau**2) + (n_states/sigma**2))
    
    # Draw gamma.
    gamma_update = random.normal(g_mean, sqrt(g_var), n_draws)
    return gamma_update
    
def muGibbsUpdate(gamma_bar, tau, n_peptides):
    # Determine the length of the arguments passed for vectorization.
    n_draws = max(1, len(gamma_bar), len(tau), len(n_peptides))
    
    # Draw mu.
    mu_update = random.normal(gamma_bar, tau/sqrt(n_peptides), n_draws)
    return mu_update

def sigmaGibbsUpdate(n_states_total, sum_of_squares, alpha, beta):
    # Determine the length of the arguments passed for vectorization.
    n_draws = max(1, len(n_states_total), len(sum_of_squares), len(alpha), len(beta))
    sigma_update = sqrt(1/random.gamma(size=n_draws, shape=(alpha + n_states_total/2), scale=1/(beta + sum_of_squares/2)))
    return sigma_update
    
def tauMarginalGibbsUpdate(n_peptides, gamma_bar, marginal_sum_of_squares, alpha, beta):
    # Determine the length of the arguments passed for vectorization.
    n_draws = max(1, len(n_peptides), len(marginal_sum_of_squares), len(alpha), length(beta))
    tau_update = sqrt(1/random.gamma(size=n_draws, shape=(alpha - 1 + (n_peptides - 1)/2), scale=1/(beta + marginal_sum_of_squares/2)))
    return tau_update

def tauGibbsUpdate(n_peptides, sum_of_squares, alpha, beta):
    # Determine the length of the argument passed for vectorization.
    n_draw = max(1, len(n_peptides), len(sum_of_squares), len(alpha), len(beta))
    tau_update = sqrt(1/random.gamma(size=n_draws, shape=(alpha+(n_peptides)/2), scale=1/(beta+sum_of_squares/2)))
    return tau_update
    
def cov2cor(matrix):
    # Convert covariance matrix to correlation matrix
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i][j] = 1
            else:
                sd_1 = sqrt(matrix[i][i])
                sd_2 = sqrt(matrix[j][j])
                matrix[i][j] = matrix[i][j] / (sd_1 * sd_2)
    return matrix
    
def lognormal_pdf(x, meanlog, sdlog):
    density = 1/(x*sqrt(2*pi*sdlog**2))*exp(-(log(x) - meanlog)^2/(2*sigma**2))
    return density 

def sign(number):
    return copysign(1.0, number)

def cube(number, arg):
    return number**2 - 4
    
