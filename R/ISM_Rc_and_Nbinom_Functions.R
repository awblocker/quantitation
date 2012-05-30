# Functions for the individual state level model with random censoring, common 
# variance and negative binomial counts for the number of states for each 
# peptide.

logit_censored_normal_pdf <- function(x, eta_0, eta_1, mu, sigma)
{
	return( 
			( (2*pi*sigma^2)^(-1/2) * exp( -(x-mu)^2 / (2*sigma^2) ) ) * 
					( 1 + exp(eta_0 + eta_1*x) )^-1
	)
}
logit_observed_normal_pdf <- function(x, eta_0= -3, eta_1= 2, mu= -1, sigma= 1)
{
	return( 
			( (2*pi*sigma^2)^(-1/2) * exp( -(x-mu)^2 / (2*sigma^2) ) ) * (exp(eta_0 + eta_1 * x) / (1 + exp(eta_0 + eta_1 * x)))
	)
}


p_censored <- function(x, eta_0, eta_1)
{
	return( 1/(1 + exp(eta_0 + eta_1*x)) )
}

# Derivitive of log density ratio.
dLogTargPropRatio <- function(x, eta_0, eta_1, mean, sd, y_hat, approx_sd, df)
{
	return	(
			-(x - mean)/sd^2 + 
					( (1 + df)*(x - y_hat) ) / (df*approx_sd^2 + (x - y_hat)^2) - 
					( eta_1 *exp(eta_0 + eta_1*x )) / (1 + exp(eta_0 + eta_1*x) )
	)
}

dCensoredIntensityLogLhood <- function(x, args)
	# First derivitive of the log likelihood for censored intensities.
	{
	# Get other arguments.
	eta_0	<- args$eta_0
	eta_1	<- args$eta_1
	mu		<- args$mu
	sigma	<- args$sigma
	# Compute the log likelihood.
	return( ( -1 + 1/(1 + exp(eta_0 + eta_1*x)) ) * eta_1 - 
			(x - mu)/(sigma^2)
			)
	}



characterizeCensoredIntensityDistribution <- function(
		eta_0, eta_1, mu, sigma, protein, peptide, TOL= 1e-5, MAX_ITER= 200,
		rand_normals)
# 1) Compute y_hat, the mode, using Halley's Method.
# 2) Compute approximate standard deviation.
# 3) Compute approximate normalizing constant, which is P(Int. Cen.) using
# 	 the Laplace Approximation.
{
	# 0) Run a few iteration of bisection to initialize Halley's method better.
	d1_args <- new.env(hash = TRUE)
	d1_args$eta_0 		<- eta_0
	d1_args$eta_1	 	<- eta_1
	d1_args$mu	 		<- mu
	d1_args$sigma		<- sigma
	x_0 <- vectorizedBisection(
			f=		dCensoredIntensityLogLhood, 
			f_args=		d1_args,
			lower= 		mu - 5*sigma,
			upper=		mu + 5*sigma,
			TOL=		1e-4,
			MAX_ITER= 	10)
#	x_0 <- mu - 2*sigma
	
	# 1) Halley's Method.
	N <- 0
	STOP_ITER <- FALSE
	x_n <- x_0#mu - 2*sigma
	while(N <= MAX_ITER & STOP_ITER == FALSE)
	{
		N <- N + 1
		x_nm1 <- x_n 
		# Evaluate the first, second and third derivitive of the log-likelihood.
		g_prime <- 	( -1 + 1/(1 + exp(eta_0 + eta_1*x_nm1)) ) * eta_1 - 
				(x_nm1 - mu)/(sigma^2)
		g_doubleprime <- (-sigma^-2 - (eta_1^2 * exp(eta_0 + eta_1 * x_nm1) ) / 
					(1 + exp(eta_0 + eta_1 * x_nm1))^2)
		g_tripleprime <- 	(2 * eta_1^3 *exp(2*eta_0 + 2*eta_1*x_nm1))/
				(1 + exp(eta_0 + eta_1*x_nm1))^3 -
				( eta_1^3 * exp(eta_0 + eta_1*x_nm1) ) / 
				(1 + exp(eta_0 + eta_1*x_nm1))^2
		x_n <- x_nm1 - 	(2*g_prime*g_doubleprime) / 
				(2*g_doubleprime^2 - g_prime*g_tripleprime)
		if(max(abs(g_prime)) < TOL)
		{
			# Save the modes.
			y_hat <- x_n
			print(paste("Halley's method iterations:", N))
			# Stop the iterations.
			STOP_ITER <- TRUE	
		}
	}
	# If too many iterations pass without convergence.
	if(STOP_ITER == FALSE)
	{
		print("Bad Initialization.")
		print(paste('Max Error:', max(abs(g_prime))))
		indx <- which(abs(g_prime) == max(abs(g_prime)))
		print(paste("Peptide:", peptide[indx])[1])
		print(paste("Protein:", protein[indx])[1])
		print(paste("Mu:", mu[indx])[1])
		y_hat <- x_n
	}
	print y_hat
	
	# 2) Approximate sd.
	approx_sd <- sqrt(-1/g_doubleprime)
	
	# 3) P(Intensity Censored), or 1/ normalizing constant.
	p_int_censored <- sqrt(2*pi/-( -sigma^-2 - 
								(eta_1^2 * exp(eta_0 + eta_1 * y_hat) ) / 
								(1 + exp(eta_0 + eta_1 * y_hat))^2 )) * 
			((1+exp(eta_0 + eta_1*y_hat))^-1) *
			( (2*pi*sigma^2)^(-1/2) * exp( -(y_hat-mu)^2 / 
								(2*sigma^2) ) )
	# Compute P(Intensity Censoring) using Monte Carlo Integration.
#	p_int_censored 	<- apply(X= as.matrix(mu), MARGIN= 1, FUN= function(x)
#				computePCenViaMonteCarloIntegration(
#						rand_normals, x, sigma, eta_0, eta_1) )
	
#	results <- list(y_hat= y_hat, approx_sd= approx_sd, 
#			p_int_censored= p_int_censored, protein= protein, peptide= peptide)
	results <- data.frame(
		protein		= protein,
		peptide		= peptide,
		y_hat		= y_hat,
		p_int_cen	= p_int_censored,
		approx_sd	= approx_sd)

	return(results)
}


computePCenViaMonteCarloIntegration <- function(random_normals, mu, sigma, 
		eta_0, eta_1)
{
	p_cens <- 1/(1 + exp(eta_0 + eta_1* (random_normals*sigma + mu)  ))
	p_int_censored <- mean(p_cens)
	return(p_int_censored)
}



dLogTargPropRatio_env <- function(x, env= dLogTargPropRatio_args)
# First derivitive of the log target-proposal ratio, using an enviroment
# to contain the additional arguments to the function.
{
	# Get the arguments from the environment.
	mu <- env$mu
	sigma <- env$sigma
	df <- env$df
	y_hat <- env$y_hat
	approx_sd <- env$approx_sd
	eta_0 <- env$eta_0
	eta_1 <- env$eta_1
	
	return	(
			-(x - mu)/sigma^2 + 
					( (1 + df)*(x - y_hat) ) / (df*approx_sd^2 + (x - y_hat)^2) - 
					( eta_1 *exp(eta_0 + eta_1*x )) / (1 + exp(eta_0 + eta_1*x) )
	)
}

vectorizedBisection <- function(f, f_args, lower, upper, TOL= 1e-10, MAX_ITER= 100)
# Find multiple roots using bisection.  f should be a function that takes
# arguments x and f_args.
{	
	# Check if the starting points are valid.
	if( sum(sign(f(lower, f_args)) == sign(f(upper, f_args))) != 0 )
	{
		warning("Invalid starting points.", call.= TRUE, immediate.= TRUE)
	}
	t <- 0
	error <- rep(1, length(lower))
	active <- rep(TRUE, length(lower))
	while(t <= MAX_ITER & max(error) > TOL)
	{
		t <- t + 1
		# Get the midpoint. 
		mid <- (lower + upper) / 2
		# Determine which values have converged.
		active[(error < TOL & active)] <- FALSE 
		error <- abs((upper - lower)/2)
		# Update values.
		f_c <- f(mid, f_args)
		lower[(sign(f_c) == sign(f(lower, f_args)) & active)] <- 
				mid[(sign(f_c) == sign(f(lower, f_args)) & active)]
		upper[(sign(f_c) == sign(f(upper, f_args)) & active)] <- 
				mid[(sign(f_c) == sign(f(upper, f_args)) & active)]
	}
	return(mid)
}


computeM <- function(eta_0, eta_1, mu, sigma, y_hat, approx_sd, 
		normalizing_cnst, df, TOL= 1e-10, MAX_ITER= 100)
# Compute the maximum of the target-proposal density ratio for the accept-
# reject sampler for censored peptide intensities.
{	
	# Store arguments for the first derivitive function.
	d1_args <- new.env(hash = TRUE)
	d1_args$eta_0 		<- eta_0
	d1_args$eta_1	 	<- eta_1
	d1_args$mu	 		<- mu
	d1_args$sigma		<- sigma
	d1_args$approx_sd	<- approx_sd
	d1_args$y_hat		<- y_hat
	d1_args$df			<- df
	
	# Make sure the starting points are the correct sign.
	left_lower <- y_hat - approx_sd
	left_upper <- y_hat - 10*TOL
	right_lower <- y_hat + 10*TOL
	right_upper <- y_hat + approx_sd
	for(i in 1:length(y_hat))
	{
		
		while(dLogTargPropRatio_env(left_lower, d1_args)[i] < 0)
		{
			left_lower[i] <- left_lower[i] - approx_sd[i]
		}	
		
		while(dLogTargPropRatio_env(left_upper, d1_args)[i] > 0)
		{
			left_upper[i] <- left_upper[i] - 10*TOL
		}	
		
		
		while(dLogTargPropRatio_env(right_lower, d1_args)[i] < 0)
		{
			right_lower[i] <- right_lower[i] + 10*TOL
		}
		
		while(dLogTargPropRatio_env(right_upper, d1_args)[i] > 0)
		{
			right_upper[i] <- right_upper[i] + approx_sd[i]
		}
	}
	# Find zeros that are less than y hat using bisection.
	left_roots <- vectorizedBisection(
			f		= dLogTargPropRatio_env,
			lower	= left_lower,
			upper	= left_upper,
			f_args	= d1_args,
			TOL		= TOL,
			MAX_ITER= MAX_ITER)
	# Find the zeros that are greater than y hat using bisection.	
	right_roots <- vectorizedBisection(
			f		= dLogTargPropRatio_env,
			lower	= right_lower,
			upper	= right_upper,
			f_args	= d1_args,
			TOL		= TOL,
			MAX_ITER= MAX_ITER)
	# Compute M.
	f_left_roots <- normalizing_cnst * logit_censored_normal_pdf(
					left_roots, eta_0, eta_1, mu, sigma) / 
			dt(x= (left_roots - y_hat)/approx_sd, df= 1)
	f_right_roots <- normalizing_cnst * logit_censored_normal_pdf(
					right_roots, eta_0, eta_1, mu, sigma) / 
			dt(x= (right_roots - y_hat)/approx_sd, df= 1) 	
	M <- apply(X= cbind(f_left_roots, f_right_roots), MARGIN= 1, FUN= max)
	results <- list(M= M, left_roots= left_roots, right_roots= right_roots)
	return(results)
}


targetProposalRatio <- function(x, mu, sigma, y_hat, approx_sd, eta_0, eta_1, 
		normalizing_cnst, df)
# Ratio of censored peptide likelihood to the proposal t density.
{
	return( (normalizing_cnst * dnorm(x, mean = mu, sd= sigma) *
						(1 + exp(eta_0 + eta_1*x))^(-1)) /
					(dt((x - y_hat)/approx_sd, df= df)))
}

censoredPeptideAcceptRejectSampler <- function(n, mu, sigma, y_hat, approx_sd,
		normalizing_cnst, eta_0, eta_1, M, df)
# Accept-reject sampler for censored peptide intensities.
{
	samples <- vector(length= n)
	n_samples <- 0
	# Draw samples until the desired number of accepted draws is achieved.
	t <- 0
	while(n_samples < n)
	{
		t <- t + 1
		proposal 	<- rt(n= 1, df = df)*approx_sd + y_hat
		unif		<- runif(n=1)
		accept_prob <- (1/M) * targetProposalRatio(proposal, mu, sigma, y_hat, 
				approx_sd, eta_0, eta_1, normalizing_cnst, df)
		# Check if the proposal is accepted.
		if(unif < accept_prob)
		{
			n_samples <- n_samples + 1
			samples[n_samples] <- proposal
		}
	}
	results <- list(samples= samples, acceptance_rate= n_samples / t)
	return(results)
}


drawNCensored <- function(p_ic, p_rc, n_obs, lmbda, r)
# Draw the number of censored peptide states.
{
	# If no states were observed for the peptide, then the support of the 
	# geometric is 1,2,3,..., but for peptides with at least one observed state
	# the support is 0,1,2,... .
	n_draws_total <- max(1, length(p_ic), length(n_obs))
	n_censored_draws <- vector(length= n_draws_total)
#	geom_p 	<- 1-(1-lmbda)*(p_rc/(p_rc + (1-p_rc)*p_ic) + p_ic*((1-p_rc)/(p_rc + (1-p_rc)*p_ic)))
	geom_p 	<- 1-(1-lmbda)*(p_rc + (1-p_rc)*p_ic)
	n_obs_g_0 <- n_obs > 0 
	n_censored_draws <- rnbinom(	n= n_draws_total, 
			p= geom_p, 
			size= n_obs + r)
	n_censored_draws <- n_censored_draws + (n_obs == 0)
#	n_draws_total <- max(1, length(p_ic), length(n_obs))
#	geom_p 	<- 1-p_ic*(1-lmbda)
#	n_censored_draws <- rgeom(n= n_draws_total, p= geom_p) + 
#						as.numeric(n_obs == 0)
	return(n_censored_draws)
}

# Block's rejection sampler.
rnCen <- function(nobs, prc, pic,lmbda, r) {
	m <- max(1, length(pic), length(nobs))
	if (length(nobs)==1)
		nobs <- rep(nobs,m)
	if (length(r)==1)
		r <- rep(r,m)
	#
	done    <- rep(FALSE, m)
	nCen    <- numeric(m)
	#
	pgeom   <- 1-(1-lmbda)*(prc+(1-prc)*pic)
	if (length(pgeom)==1)
		pgeom <- rep(pgeom,m)
	#
	bound <- rep(1, m)
#	bound[r<1] <- bound[r<1]*nobs[r<1]/(nobs[r<1]+r[r<1]-1)
	bound[r<1] <- bound[r<1]*(nobs[r<1]+r[r<1]-1)/nobs[r<1]
	#
	nIter <- 0
	while(sum(done)<m) {
		prop    <- rnbinom(m-sum(done), nobs[!done]+r[!done], pgeom[!done])
		#
		u       <- runif(m-sum(done))
		pAccept <- (nobs[!done]+prop)/
				(nobs[!done]+prop+r[!done]-1)*
				bound[!done]
		pAccept[nobs[!done]==0] <- 1
		#
		replace <- which(!done)
		nCen[replace[u<pAccept]] <- prop[u<pAccept]
		done[replace] <- u<pAccept
		#
		nIter       <- nIter + 1
	}
	nCen <- nCen + (nobs==0)
	return(nCen)
}



drawNCensoredMH <- function(p_ic, n_obs, lmbda, n_cen_tm1)
# Draw n censored using Metropollis-Hastings.
{
#	geom_p <- 1-p_ic*(1-lmbda)
	geom_p <- lmbda
	n_draws <- max(1, length(p_ic), length(n_obs))
	# Draw n cens.
	n_cen_star <- rgeom(n= n_draws, prob= geom_p) +
			as.numeric(n_obs == 0)
	# Calculate the log of the importance ratio minus the prior ratio.
	LPR_m_LIR <- (n_cen_star - n_cen_tm1) * log(p_ic)
	logU	<- log(runif(n= n_draws))
	# Set n_cen(t) acording the the MH update rule. 
	update <- logU < LPR_m_LIR
	n_cen_t <- n_cen_tm1
	n_cen_t[update] <- n_cen_star[update] 
	return(n_cen_t)
}


drawCensoredIntensities <- function(n, mu, sigma, y_hat, approx_sd,
		p_ic, p_rc, eta_0, eta_1, df, protein, peptide)
# Draws the intensities of censored peptides.
{
	censored_intensities <- data.frame(matrix(nrow= sum(n), ncol= 6))
	colnames(censored_intensities) <- c('Protein', 'Peptide', 'State', 
			'Intensity', 'Observed', "W")
	# First, get the maximum of the target, proposal ratio.
	computeMResults <- computeM(
			eta_0				= eta_0,
			eta_1				= eta_1,
			mu					= mu,
			sigma				= sigma,
			y_hat				= y_hat,
			approx_sd			= approx_sd, 
			normalizing_cnst	= 1/p_ic,
			df					= df,
			TOL					= 1e-10,
			MAX_ITER			= 100)
	M <- computeMResults$M								
	
	# Iterate through the list of peptides.
	cen_count <- 0
	for(i in 1:length(mu))
	{
#		print(paste('Peptide:', i))
		if(n[i] > 0)
		{
			# Determine the indices to save the intensities in.
			start 	<- cen_count + 1
			end 	<- cen_count + n[i]
			# Draw the number of randomly censored peptides.
			n_rand_cen <- rbinom(n = 1, size= n[i], p= p_rc / (p_rc + (1-p_rc) * p_ic[i]))
			randCenInts <- rnorm(n= n_rand_cen, mean= mu[i], sd= sigma[i])
			#Draw the intensity-censored peptide intensities.
			intCenIntsResults <- censoredPeptideAcceptRejectSampler(
					n					= n[i] - n_rand_cen,
					mu					= mu[i],
					sigma				= sigma[i],
					y_hat				= y_hat[i],
					approx_sd			= approx_sd[i],
					normalizing_cnst	= 1/p_ic[i],
					eta_0				= eta_0,
					eta_1				= eta_1,
					M					= M[i],
					df					= df)
			int_cen_ints <- intCenIntsResults$samples
			# Save the latent random censoring indicators and the censored intensities.
			censored_intensities[(start:end),'Protein'] 	<- protein[i]
			censored_intensities[(start:end),'Peptide'] 	<- peptide[i]
			censored_intensities[(start:end),'State']		<- 'Censored_State'
			censored_intensities[(start:end),'Observed']	<- FALSE
			censored_intensities[(start:end),'Intensity'] 	<- c(int_cen_ints, randCenInts)
			censored_intensities[(start:end),'W']			<- c(rep(0, n[i] - n_rand_cen), rep(1, n_rand_cen))
			cen_count <- cen_count + n[i]
			# End of the "if(n[i] > 0)" loop.
		}	
	}
	return(censored_intensities)
}
#drawCensoredIntensities <- Vectorize(drawCensoredIntensities)


drawCensoredIntensitiesFast <- function(n, mu, sigma, y_hat, approx_sd,
		p_ic, p_rc, eta_0, eta_1, df, protein, peptide)
# Draws the intensities of censored peptides.
{
	censored_intensities <- data.frame(matrix(nrow= sum(n), ncol= 6))
	colnames(censored_intensities) <- c('Protein', 'Peptide', 'State', 
			'Intensity', 'Observed', "W")
	# First, get the maximum of the target, proposal ratio.
	computeMResults <- computeM(
			eta_0				= eta_0,
			eta_1				= eta_1,
			mu					= mu,
			sigma				= sigma,
			y_hat				= y_hat,
			approx_sd			= approx_sd, 
			normalizing_cnst	= 1/p_ic,
			df					= df,
			TOL					= 1e-10,
			MAX_ITER			= 100)
	M <- computeMResults$M								
	# Get the number of peptides that we need to draw intensities for.
	n_peps <- max(1, length(mu), length(sigma), length(n), length(approx_sd), 
			length(y_hat))
	n_draws_total <- sum(n)
	# Create a dataframe of arguments to use in the accept-reject sampler.
	accept_reject_df <- data.frame(matrix(nrow= sum(n), ncol= 11))
	colnames(accept_reject_df) <- c('Intensity', 'mu', 'sigma', 'y_hat', 
			'approx_sd', 'p_ic', 'M', "W", 'Protein', "Peptide",
			"accept_probs")
	accept_reject_df$mu <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(mu[x], n[x]) ) )
	accept_reject_df$sigma <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(sigma[x], n[x]) ) )
	accept_reject_df$y_hat <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(y_hat[x], n[x]) ) )
	accept_reject_df$approx_sd <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(approx_sd[x], n[x]) ) )
	accept_reject_df$normalizing_cnst <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(1/p_ic[x], n[x]) ) )
	accept_reject_df$p_ic <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(p_ic[x], n[x]) ) )
	
	accept_reject_df$M <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(M[x], n[x]) ) )
	accept_reject_df$sigma <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(sigma[x], n[x]) ) )
	accept_reject_df$Protein <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(protein[x], n[x]) ) )
	accept_reject_df$Peptide <- unlist(sapply(1:n_peps, FUN= function(x) 
						rep(peptide[x], n[x]) ) )
	
	# Draw the random censoring indicators.
	accept_reject_df$W <- rbinom(n= n_draws_total, size= 1, 
			prob = p_rc / (p_rc + (1-p_rc)*accept_reject_df$p_ic) )
	# Draw the randomly censored intensities.
	n_rand_censored <- sum(accept_reject_df$W)
	if(n_rand_censored > 0)
		{
		accept_reject_df[which(accept_reject_df$W == 1), 'Intensity'] <- rnorm(
				n= n_rand_censored, 
				mean = accept_reject_df[which(accept_reject_df$W == 1), 'mu'],
				sd= accept_reject_df[which(accept_reject_df$W == 1), 'sigma'])
		}	
	# Draw the intensity-censored intensities using the accept reject sampler.
	# Create an indicator for active draws. 
	active <- matrix(nrow= n_draws_total, ncol= 1, data= TRUE)
	active[which(accept_reject_df$W == 1)] <- FALSE
	while(sum(active) > 0)
	{
		n_active <- sum(active)
		accept_reject_df[active, 'Intensity'] <- rt(n= n_active, df= df) *
				accept_reject_df[active, 'approx_sd'] +
				accept_reject_df[active, 'y_hat']
		unifs <- runif(n= n_draws_total)
		accept_reject_df[active, 'accept_probs'] <- (1/accept_reject_df[active, 'M']) * 
				targetProposalRatio(
						accept_reject_df[active, 'Intensity'],
						accept_reject_df[active, 'mu'],
						accept_reject_df[active, 'sigma'],
						accept_reject_df[active, 'y_hat'],
						accept_reject_df[active, 'approx_sd'],
						eta_0, eta_1, 
						1/accept_reject_df[active, 'p_ic'],
						df)
		active[(accept_reject_df$accept_probs > unifs) & active] <- FALSE  
		
	}
	censored_intensities$Protein 		<- accept_reject_df$Protein
	censored_intensities$Peptide		<- accept_reject_df$Peptide
	censored_intensities$State			<- 'Censored_State'
	censored_intensities$Observed		<- FALSE
	censored_intensities$Intensity 		<- accept_reject_df$Intensity
	censored_intensities$W				<- accept_reject_df$W
	
#	return(list(censored_intensities= censored_intensities,
#					accept_reject_df = accept_reject_df))
	return(censored_intensities)
	
}


computeMeanAndSd <- function(x)
# Function to compute the mean and standard deviation of a vector.
{
	n		<- length(x)
	x_bar	<- mean(x)
	sd		<- sqrt(sum((x - x_bar)^2) / (n -1 ))
	return(list(x_bar= x_bar, sd= sd))
}

updateEtaMH <- function(verbose= FALSE)
# Update the intensity-censoring GLM coefficients.
{
	eta_model 	<- speedglm(
			data= all_peptides_df[which(all_peptides_df$W != 1),],
			formula= Observed ~ Intensity, family= binomial(link= 'logit'))
	eta_hat		<- eta_model$coef
	eta_hat_cov	<- vcov(eta_model)
	eta_star	<- rmvnorm(n= 1, mean= eta_hat, sigma= eta_hat_cov)
	# MH step.
	p_eta_star	<- 1/(1 + exp(-1*(eta_star[1] + eta_star[2] * 
									all_peptides_df[which(all_peptides_df$W != 1),'Intensity'])))
	p_eta_prev	<- 1/(1 + exp(-1*(eta[t-1,1] + eta[t-1,1] *
									all_peptides_df[which(all_peptides_df$W != 1),'Intensity'])))
	lpr_eta 	<- sum(
			dbinom(all_peptides_df[which(all_peptides_df$W != 1),'Observed'],1,
							p_eta_star, log= TRUE) - 
					dbinom(all_peptides_df[which(all_peptides_df$W != 1),'Observed'],1,
							p_eta_prev, log= TRUE))
	lir_eta 	<- dmvnorm(
					x= eta_star, mean= eta_hat,sigma= eta_hat_cov,log= TRUE)-
			dmvnorm(x= eta[t-1,], mean= eta_hat, sigma= eta_hat_cov, log= TRUE)
	log_r_eta	<- lpr_eta - lir_eta
	log_u_eta	<- log(runif(n= 1))
	mh_update_eta <- log_u_eta < log_r_eta
	if(mh_update_eta == TRUE)
	{
		eta[t,] <- eta_star
	}else{	
		eta[t,] <- eta[t-1,]
	}
	if(verbose)
	{
		print(paste('Eta MH Update:', mh_update_eta))
		print(summary(eta_model))
	}
}




drawCensoredIntensitiesOneFunction <- function(eta_0, eta_1, mu, sigma, lambda,
		p_rc, n_obs, df, protein, peptide, TOL, MAX_ITER)
# Combine all of the steps to draw the censored peptide intensities into one
# function.
{
	# 1) Compute y hat using Halley's Method.
	N <- 0
	STOP_ITER <- FALSE
	x_n <- mu
	while(N <= MAX_ITER & STOP_ITER == FALSE)
	{
		N <- N + 1
		x_nm1 <- x_n 
		# Evaluate the first, second and third derivitive of the log-likelihood.
		g_prime <- 	( -1 + 1/(1 + exp(eta_0 + eta_1*x_nm1)) ) * eta_1 - 
				(x_nm1 - mu)/(sigma^2)
		g_doubleprime <- (-sigma^-2 - (eta_1^2 * exp(eta_0 + eta_1 * x_nm1) ) / 
					(1 + exp(eta_0 + eta_1 * x_nm1))^2)
		g_tripleprime <- 	(2 * eta_1^3 *exp(2*eta_0 + 2*eta_1*x_nm1))/
				(1 + exp(eta_0 + eta_1*x_nm1))^3 -
				( eta_1^3 * exp(eta_0 + eta_1*x_nm1) ) / 
				(1 + exp(eta_0 + eta_1*x_nm1))^2
		x_n <- x_nm1 - 	(2*g_prime*g_doubleprime) / 
				(2*g_doubleprime^2 - g_prime*g_tripleprime)
		if(max(abs(x_n - x_nm1)) < TOL)
		{
			# Save the modes.
			y_hat <- x_n
			print(paste("Halley's method iterations:", N))
			# Stop the iterations.
			STOP_ITER <- TRUE	
		}
	}
	# If too many iterations pass without convergence.
	if(STOP_ITER == FALSE)
	{
		print("Bad Initialization.")
		print(paste('Max Error:', max(abs(x_n - x_nm1))))
		y_hat <- x_n
	}
	
	# 2) Approximate sd.
	approx_sd <- sqrt(-1/g_doubleprime)
	
	# 3) P(Intensity Censored), or 1/ normalizing constant.
	p_ic <- sqrt(2*pi/-( -sigma^-2 - 
								(eta_1^2 * exp(eta_0 + eta_1 * y_hat) ) / 
								(1 + exp(eta_0 + eta_1 * y_hat))^2 )) * 
			((1+exp(eta_0 + eta_1*y_hat))^-1) *
			( (2*pi*sigma^2)^(-1/2) * exp( -(y_hat-mu)^2 / 
								(2*sigma^2) ) )
	normalizing_cnst <- 1/p_ic
	
	# 4) Determine the number of censored peptides.
	# If no states were observed for the peptide, then the support of the 
	# geometric is 1,2,3,..., but for peptides with at least one observed state
	# the support is 0,1,2,... .
	p_censored <- (p_rc + (1-p_rc) * p_ic)*lambda
	# First, draw a random geometric with support 
	n_censored <- rgeom(n= length(p_ic), prob= 1 - p_censored)
	# Add one if no states were observed.
	n_censored <-  n_censored + (n_obs == 0)
	if(n_censored > 0)
	{
		# 5) Draw the latent random censoring indicators.
		p_rand_cen 	<- p_rc / ( p_rc + (1-p_rc)*p_ic )
		n_rand_cen 	<- rbinom(n= 1, prob= p_rand_cen, size= n_censored)
		n_int_cen	<- n_censored - n_rand_cen
		
		# 6) Draw the censored intensities.
		# Draw the randomly censored intensities.	
		if(n_rand_cen > 0)
		{
			rand_cen_ints <- rnorm(n= n_rand_cen, mean= mu, sd= sigma)	
		}else{
			rand_cen_ints <- c()
		}
		# Draw the intensity-censored intensities. 
		if(n_int_cen > 0)
		{
			# Compute the maximum of the target-proposal likelihood ratio.
			# Store arguments for the first derivitive function.
			d1_args <- new.env(hash = TRUE)
			d1_args$eta_0 		<- eta_0
			d1_args$eta_1	 	<- eta_1
			d1_args$mu	 		<- mu
			d1_args$sigma		<- sigma
			d1_args$approx_sd	<- approx_sd
			d1_args$y_hat		<- y_hat
			d1_args$df			<- df
			
			# Find zeros that are less than y hat using bisection.
			left_lower <- y_hat - 4*approx_sd
			left_upper <- y_hat - .1
			left_roots <- vectorizedBisection(
					f		= dLogTargPropRatio_env,
					lower	= left_lower,
					upper	= left_upper,
					f_args	= d1_args,
					TOL		= TOL,
					MAX_ITER= MAX_ITER)
			# Find the zeros that are greater than y hat using bisection.
			right_lower <- y_hat + 4*approx_sd
			right_upper <- y_hat + .1
			right_roots <- vectorizedBisection(
					f		= dLogTargPropRatio_env,
					lower	= right_lower,
					upper	= right_upper,
					f_args	= d1_args,
					TOL		= TOL,
					MAX_ITER= MAX_ITER)
			# Compute M.
			f_left_roots <- normalizing_cnst * logit_censored_normal_pdf(
							left_roots, eta_0, eta_1, mu, sigma) / 
					dt(x= (left_roots - y_hat)/approx_sd, df= 1)
			f_right_roots <- normalizing_cnst * logit_censored_normal_pdf(
							right_roots, eta_0, eta_1, mu, sigma) / 
					dt(x= (right_roots - y_hat)/approx_sd, df= 1) 	
			M <- apply(X= cbind(f_left_roots, f_right_roots), MARGIN= 1, FUN= max)
			
			# Accept reject sampler.
			# Store arguments for the first derivitive function.
			d1_args <- new.env(hash = TRUE)
			d1_args$eta_0 		<- eta_0
			d1_args$eta_1	 	<- eta_1
			d1_args$mu	 		<- mu
			d1_args$sigma		<- sigma
			d1_args$approx_sd	<- approx_sd
			d1_args$y_hat		<- y_hat
			d1_args$df			<- df
			
			# Find zeros that are less than y hat using bisection.
			left_lower <- y_hat - 4*approx_sd
			left_upper <- y_hat - .1
			left_roots <- vectorizedBisection(
					f		= dLogTargPropRatio_env,
					lower	= left_lower,
					upper	= left_upper,
					f_args	= d1_args,
					TOL		= TOL,
					MAX_ITER= MAX_ITER)
			# Find the zeros that are greater than y hat using bisection.
			right_lower <- y_hat + 4*approx_sd
			right_upper <- y_hat + .1
			right_roots <- vectorizedBisection(
					f		= dLogTargPropRatio_env,
					lower	= right_lower,
					upper	= right_upper,
					f_args	= d1_args,
					TOL		= TOL,
					MAX_ITER= MAX_ITER)
			# Compute M.
			f_left_roots <- normalizing_cnst * logit_censored_normal_pdf(
							left_roots, eta_0, eta_1, mu, sigma) / 
					dt(x= (left_roots - y_hat)/approx_sd, df= 1)
			f_right_roots <- normalizing_cnst * logit_censored_normal_pdf(
							right_roots, eta_0, eta_1, mu, sigma) / 
					dt(x= (right_roots - y_hat)/approx_sd, df= 1) 	
			M <- apply(X= cbind(f_left_roots, f_right_roots), MARGIN= 1, FUN= max)
			
			# Accept reject sampler.
			samples <- vector(length= n_int_cen)
			n_samples <- 0
			# Draw samples until the desired number of accepted draws is achieved.
			t <- 0
			while(n_samples < n_int_cen)
			{
				t <- t + 1
				proposal 	<- rt(n= 1, df = df)*approx_sd + y_hat
				unif		<- runif(n=1)
				accept_prob <- (1/M) * targetProposalRatio(proposal, mu, sigma, y_hat, 
						approx_sd, eta_0, eta_1, normalizing_cnst, df)
				# Check if the proposal is accepted.
				if(unif < accept_prob)
				{
					n_samples <- n_samples + 1
					samples[n_samples] <- proposal
				}
			}
			# Save the censored intensity draws.
			int_cen_ints <- samples
		}else{
			int_cen_ints <- c()
			# End of the "if(n_int_cen > 0)" loop.
		}
		# Save the censored intensity draws.
		censored_intensities				<- data.frame(matrix(nrow= n_censored,
						ncol= 6))
		colnames(censored_intensities) <- c('Protein', 'Peptide', 'State', 
				'Intensity', 'Observed', 'W')						
		censored_intensities[,'Protein'] 	<- protein
		censored_intensities[,'Peptide'] 	<- peptide
		censored_intensities[,'State']		<- 'Censored_State'
		censored_intensities[,'Observed']	<- FALSE
		w	 	<- c(rep(1, n_int_cen), rep(0, n_rand_cen))
		ints	<- c(rand_cen_ints, int_cen_ints)
		censored_intensities[,'W']	 		<- w
		censored_intensities[,'Intensity'] 	<- ints
		# Return the censored intensities.
		return(censored_intensities)
	}
	# End if(n_censored > 0) loop.
}
drawCensoredIntensitiesOneFunction <- Vectorize(drawCensoredIntensitiesOneFunction)


MORE_LETTERS <- vector(length= length(LETTERS) * 10)
count <- 0
for(letter in LETTERS)
{
	for(number in 1:9)
	{
		count <- count + 1
		new_letter <- paste(letter, number, sep= '')
		MORE_LETTERS[count] <- new_letter
	}
}



# Functions for the negative binomial parameter joint estimation.
profLikNbinom <- function(r, x, r_prior= c(1,0))
	# Profile likelihood for r, with lambda fixed at its MLE.
	# Note: This assumes that x = 1, 2,... instead of the usual 
	# support of x = 0, 1, 2, 3... .
	{
	n <- length(x)
#	p_hat <- (r + 1)/(mean(x)+r + 1)
	p_hat <- r/(mean(x) + r - 1)
	return(
		sum(dnbinom(x-1, prob= p_hat, size=r, log=TRUE))+
				(r_prior[1]-1)*log(r) - r_prior[2]*exp(log(r)) #+
#				log(r) + log(p_hat) + log(1-p_hat)
	)
	} 

# Setup actual log-likelihood
loglikNbinom <- function(logitP, logR, x, r_prior= c(1,0))
	# Negative binomial log likelihood.
	# Note: This assumes that x = 1, 2,... instead of the usual 
	# support of x = 0, 1, 2, 3... .
	{
	p <- 1/(1+exp(-logitP))
	r <- exp(logR)
	return(
			sum(dnbinom(x-1,prob=p,size=r,log=TRUE)) + 
					(r_prior[1]-1)*logR - r_prior[2]*exp(logR) #+
#					log(p*(1-p)) + logR
	)
	}
	
	
# Functions for computing the information matrix for the reparameterzed model
# parameters, specifically logit(p) and log(r).

d2P <- function(p,r,n,x) 		
			( (-n*r)/p^2 - (sum(x) - n)/(1-p)^2)*# - 1/p^2 - 1/(1-p)^2) *
			(p * (1-p))^2 +
			((n*r)/p - (sum(x) - n)/(1-p))*# + 1/p -1/(1-p)) *
			(exp(log(p/(1-p)))*(1-exp(log(p/(1-p))))) / (1+exp(log(p/(1-p))))^3


d2R <- function(p,r,n,x, r_prior= c(1,0)) 		
			(sum(digamma( x + r - 1)) - n * digamma(r) + n * log(p) + 
						(r_prior[1] - 1)/r - r_prior[2]) * r +
			(sum(trigamma(x + r - 1)) -n*trigamma(r) - 
				(r_prior[1]-1)/r^2 ) * r^2

#d2P <- function(p, r, n, x)
#	{	
#	( (-n * (r + 1))/p^2 - sum(x)/(1-p)^2 ) * (p * (1-p))^2 +
#	( (n * (r+1))/p  - sum(x)/(1-p)) * ( (p/(1-p)) *(1-p/(1-p)) ) / (1 + p/(1-p))^3
#	}
#	
#d2R <- function(p, r, n, x, r_prior)
#	{
#	( sum(trigamma(x + r - 1)) - n*trigamma(r) - n/r^2 - (r_prior[1] -1)/r^2)* r^2 +
#	( sum(digamma(x + r -1)) - n*digamma(r) + n*log(p) + n/r + (r_prior[1] -1)/r - r_prior[2]) * r
#	}

crossPR <- function(p,r,n)	(p*(1-p)) * (r) * (n/p)

# Setup information matrix
infoMat <- function(theta,x, r_prior= c(1,0)) {
	p <- exp(theta[1])/(1+exp(theta[1]))
	r <- exp(theta[2])
	n <- length(x)
	-matrix(
			c( 	d2P(p,r,n,x),
					crossPR(p,r,n),
					crossPR(p,r,n),
					d2R(p,r,n,x, r_prior)
			)
			,2,2)
}


computeNbinomMLEs <- function(x, r_prior = c(0,1))
	# Maximize the profile likelihood of r to obtain the MLEs of logit(p) and 
	# log(r).  Also computes their covariance matrix using the negative inverse
	# of the observed information matrix.
	{	
	# Compute the MLEs.
	rMLE <- optimize( profLikNbinom, c(0.01, 50), x=x, r_prior= r_prior,
			maximum=TRUE )$maximum
	nbinomMLEs <- c((rMLE)/(mean(x)+rMLE - 1), rMLE)
	# Compute the covariance matrix.
	covMat <- solve(infoMat(c(log(nbinomMLEs[1]/(1-nbinomMLEs[1])), 
							log(nbinomMLEs[2])), x, r_prior = r_prior))
	results <- list(logitLambdaMLE= log(nbinomMLEs[1]/ (1-nbinomMLEs[1])),
					logRMLE= log(nbinomMLEs[2]),
					covMat= covMat)
	return(results)
	}


lambdaAndRMH <- function(n_by_pep, lambda_tm1, r_tm1, r_prior= c(1,0))
	# Compute the MLE for the negative binomial parameters.  Then draw proposals
	# for (logit(lambda), log(r)) using the normal approximation.  Finally, perform 
	# the MH step and return (lambda[t], r[t]).
	{
 	# Compute the MLEs for logit(p) and log(r).
	MLEs <- computeNbinomMLEs(n_by_pep, r_prior = r_prior)
	logitLambdaMLE 	<- MLEs$logitLambdaMLE
	logRMLE 		<- MLEs$logRMLE
	covMat			<- MLEs$covMat * 1000
	print('covMat:')
	print(covMat)
	print('corMat:')
	print(cov2cor(covMat))
	# Draw proposals for logit(lambda) and log(r).
	proposals <- rmvnorm(n= 1, mean= c(logitLambdaMLE, logRMLE), sigma= covMat)

	print(paste('r MLE:', exp(logRMLE)))
	print(paste('r proposal:', exp(proposals[2])))
	print(paste('lambda MLE:', 1/(1+exp(-logitLambdaMLE))))
	print(paste('lambda proposal:', 1/(1+exp(-proposals[1]))))
	
	# Metropolis-Hasting.
	lpr <- 	loglikNbinom(logitP= proposals[1], logR= proposals[2], x= n_by_pep, 
					r_prior = r_prior) -
			loglikNbinom(	logitP= log(lambda_tm1/ (1-lambda_tm1)),
							logR= log(r_tm1), x= n_by_pep,
					r_prior = r_prior)
	print(paste('lpr:', lpr))
#	lpr <- sum( 	lgamma(n_by_pep + exp(proposals[2]) - 1) + lgamma(r_tm1) -
#					lgamma(n_by_pep + r_tm1 - 1) - lgamma(exp(proposals[2])) +
#					exp(proposals[2])*log(1/(1+exp(-proposals[1]))) -
#					r_tm1*log(lambda_tm1)
#				)
	prop_r <- exp(proposals[2])
	prop_p <- 1/(1+ exp(-proposals[1]))
	
	lir <- 	dmvnorm(x= proposals, mean= c(logitLambdaMLE, logRMLE), 
					sigma= covMat, log= TRUE)- log(prop_r) - log(prop_p) - log(1-prop_p) -
			(dmvnorm(x= c(log(lambda_tm1/ (1-lambda_tm1)), log(r_tm1)),
					mean= c(logitLambdaMLE, logRMLE), 
					sigma= covMat, log= TRUE) - log(r_tm1) - log(lambda_tm1) - log(1-lambda_tm1))
	print(paste('lir:', lir))
	# MH Update.
	print(paste('lpr - lir:',lpr - lir))
	log_u <- log(runif(n=1))
	if( log_u < (lpr - lir)   )
		{
		lambda_t 	<- 1/(1 + exp(-proposals[1]))
		r_t			<- exp(proposals[2])
		}else{
		lambda_t 	<- lambda_tm1
		r_t			<- r_tm1	
		}
	return(c(lambda_t, r_t))
	}

	
# Use the profile likelihood to draw log(r), then use the conditional draw for
# lambda|log(r) to get lambda.  Then do the metropolis step jointly.

# Set up the information matrix for log(r).
logRInformation <- function(r, x, r_prior= c(1,0))
	# Information matrix for log(r), using the negative binomial profile 
	# likelihood.
	{
	n <- length(x)
	n_bar <- mean(x)
	d1r_profile <- sum(digamma(x + r -1)) - n*digamma(r) + 
			n*( (n_bar-1)/(n_bar+r-1) + log(r/(n_bar+r-1)) ) -
			sum(x-1)/(n_bar+r-1) + (r_prior[1]-1)/r - r_prior[2]
	d2r_profile <- sum(trigamma(x+r-1)) - n*trigamma(r) +
			n*(n_bar-1)^2/(r*(n_bar+r-1)^2) +
			sum(x-1)/(n_bar+r-1)^2 - (r_prior[1] -1)/r^2
	iMat_profile <- (d1r_profile*r + d2r_profile*r^2)
#	iMat_profile <- d2r_profile
	return(iMat_profile)
	}

	
rLambdaMH2 <- function(n_by_pep, lambda_tm1, r_tm1, r_prior= c(1,0))
	{
	# Get r MLE.
	rMLE <- optimize( profLikNbinom, c(0.01, 50), x=n_by_pep, r_prior= r_prior,
			maximum=TRUE )$maximum
	logRMLE <- log(rMLE)
	logRMLEVar <-  1/(-logRInformation(r= rMLE, x= n_by_pep, r_prior= r_prior))
#	print(paste('r mle:', rMLE))
	# Draw the proposal for log(r).
	r_star <- rlnorm(n = 1, meanlog = logRMLE, sdlog = sqrt(logRMLEVar))
#	print(paste('r star:', r_star))
	# Draw lambda|r_star.
	n_peptides_total 	<- length(n_by_pep)
	n_states_total		<- sum(n_by_pep)
	
	lambda_star <- rbeta(n = 1, shape1 = r_star*n_peptides_total + 1, 
			shape2= n_states_total- n_peptides_total + 1)
#	print(paste('lambda star:', lambda_star))
	# Compute the prior ratio.
	lpr <- 	sum(dnbinom(x = n_by_pep-1, prob = lambda_star, size = r_star, log= TRUE)) +
			(r_prior[1]-1)*log(r_star) - r_prior[2]*r_star -
			(sum(dnbinom(x = n_by_pep-1, prob = lambda_tm1, 	size = r_tm1, log= TRUE)) +
			(r_prior[1]-1)*log(r_tm1) - r_prior[2]*r_tm1)
#	print(paste('lpr:', lpr))
	# Compute the importance ratio.
	lir <-	dlnorm(x= r_star, meanlog = logRMLE, sdlog = sqrt(logRMLEVar), log= TRUE) +
			dbeta(x= lambda_star, shape1= r_star*n_peptides_total + 1, 
					shape2= n_states_total - n_peptides_total + 1, log= TRUE) -
			dlnorm(x= r_tm1, meanlog = logRMLE, sdlog = sqrt(logRMLEVar), log= TRUE) -
			dbeta(x= lambda_tm1, shape1= r_tm1*n_peptides_total + 1, 
					shape2= n_states_total - n_peptides_total + 1, log= TRUE)
#	print(paste('lir:', lir))
#	print(paste('lpr-lir:', lpr-lir))
	# MH Step.
	log_u <- log(runif(n=1))
	if(log_u < (lpr - lir))
	{
		lambda_t	<- lambda_star 
		r_t 		<- r_star
	}else{
		lambda_t	<- lambda_tm1
		r_t			<- r_tm1
	}
	return(c(lambda_t, r_t))
	}
	

#--- Hierarchical model updates. ---#


# Gamma hyper prior update.
profLikGamma <- function(a, x)
	# Profile log likelihood of the shape parameter of a gamma distribution, with
	# mean shape/rate.
	{
	b_hat <- (a - 1/length(x))/mean(x)
#	return(sum(dgamma(x, shape= a, rate= b_hat, log=TRUE)) - log(b_hat) - log(a))
	return(sum(dgamma(x, shape= a, rate= b_hat, log=TRUE)) - log(b_hat)  + 
			dlnorm(x= a, meanlog = 2.65, sdlog = .652, log= TRUE) )
	}
logLikelihoodGamma <- function(logA, logB, x)
	# Log likelihood for a gamma with the shape and rate log transformed.
	{
	return(sum(dgamma(x,exp(logA),exp(logB),log=TRUE)))
	}


# Setup information matrix
#infoMatGamma <- function(theta,x)
#	# Information matrix for log transformed gamma parameters.
#	{
#	a <- exp(theta[1])
#	b <- exp(theta[2])
#	n <- length(x)
#	iMat <- -matrix(c(a*(n*log(b)-n*digamma(a)+sum(log(x)))-a*a*n*trigamma(a),a*n,
#					a*n,-b*sum(x)),2,2)
#	return(iMat)
#	}

infoMatGamma <- function(theta,x)
# Information matrix for log transformed gamma parameters.
{
	a <- exp(theta[1])
	b <- exp(theta[2])
	n <- length(x)

#	iMat <- -matrix(c(a*(n*log(b)-n*digamma(a)+sum(log(x) + ( (.652^2 - log(a) -2*2.65)/(.652^2 * a) ) ))+
	iMat <- -matrix(c(
					a*(n*log(b)-n*digamma(a)+sum(log(x)) + ( (-1*(.652^2) - log(a) + 2.65)/(.652^2 * a) ) )+
							a*a*(-n*trigamma(a) + ( (-1-2.65 +.652^2 + log(a))/(a^2*(.652)^2) )),
					a*n,
					a*n,
					-b*sum(x)),2,2)
	return(iMat)
}



dmvlnorm <- function(x, logmean, logsigma, LOG)
	{
	p <- length(logmean)
	L_x <- (2*pi)^(-p/2)*det(logsigma)^(-1/2) * prod(1/x) %*% 
			exp((-1/2)* (log(x)-logmean)%*%solve(logsigma)%*%t(log(x)-logmean))
	if(LOG==TRUE)
		{
		L_x <- log(L_x)
		}
	return(L_x)
	}

Logmvlnorm <- function(x, logmean, logsigma)
	{
		
		L_x <- 	dmvnorm(x = log(x), mean = logmean, sigma = logsigma, log= TRUE) -
				sum(log(x))
		return(L_x)
	}
	
	
varianceHyperparametersMHUpdate <- function(variances, alpha_tm1, beta_tm1)
	# Metropolis-Hastings update for the variance hyper parameters, using the 
	# normal approximation of the log of their MLEs.
	{
	precisions <- 1/variances
	# Compute the MLE for the shape parameter using the profile likelihood.
	alphaMLE <- optimize( profLikGamma, c(0.01, 10000), x=precisions, maximum=TRUE )$maximum
	logAlphaMLE <- log(alphaMLE)
	# Use the MLE for the shape to compute the MLE for the rate.
	betaMLE <- (alphaMLE - 1/length(variances))/mean(precisions)
	print(paste('alpha mle, beta mle:', alphaMLE, betaMLE))
	logBetaMLE <- log(betaMLE)
	#Calculate the observed information matrix for the log transformed
	# parameters.
	infoMat <- infoMatGamma(theta= c(logAlphaMLE, logBetaMLE), x= precisions)
	covMat	<- solve(infoMat)
	print('Info matrix:')
	print(infoMat)
	print('Covariance Matrix:')
	print(covMat)
	print('Correlation Matrix:')
	print(cov2cor(covMat))
	# Draw proposals for alpha and beta using the log normal centered around 
	# their logMLEs.
	proposals	<- exp(rmvnorm(n= 1, mean = c(logAlphaMLE, logBetaMLE), sigma = covMat))
	print(paste('alpha star, beta star:', proposals[1], proposals[2]))
	alphaStar	<- proposals[1]
	betaStar	<- proposals[2] 
	# Compute the prior ratio.
	lpr <- 	sum(dgamma(x= precisions, shape= alphaStar, rate= betaStar, log= TRUE)) -
#			log(betaStar) - log(alphaStar) -
			log(betaStar) + dlnorm(x= alphaStar, meanlog = 2.65, sdlog = .652, log= TRUE) -
			sum(dgamma(x= precisions, shape= alpha_tm1, rate= beta_tm1, log= TRUE)) +
#			log(beta_tm1) + log(alpha_tm1)
			log(betaStar) - dlnorm(x= alpha_tm1, meanlog = 2.65, sdlog = .652, log= TRUE)
	print(paste('lpr:', lpr))
	lir	<-	Logmvlnorm(x= proposals, logmean= c(logAlphaMLE, logBetaMLE), logsigma= covMat) -
			Logmvlnorm(x= c(alpha_tm1, beta_tm1),logmean= c(logAlphaMLE, logBetaMLE), logsigma= covMat)
	print(paste('lir:', lir))
	print(paste('lpr-lir:', lpr-lir))
	# MH Step.
	log_u <- log(runif(n=1))
	print(paste('MH Update:', (log_u < (lpr-lir))))
	if(log_u < (lpr-lir))
		{
		alpha_t <- alphaStar
		beta_t	<- betaStar
		}else{
		alpha_t <- alpha_tm1
		beta_t	<- beta_tm1
		}
	return(c(alpha_t, beta_t))	
	}
	

gammaGibbsUpdate <- function(mu, tau, sigma, y_bar, n_states)
	{
	# Determine the length of the arguments passed for vectorization.
	n_draws <- max(1, length(mu), length(tau), length(sigma), length(y_bar), length(n_states))
	
	# Comput the mean and variance of the conditional distribution of gamma|... .
	g_mean 	<- ( (mu/tau^2) + (y_bar/(sigma^2/n_states) ) ) / ( (1/tau^2) + (n_states/sigma^2) )
	g_var	<-  1/ ( (1/tau^2) + (n_states/sigma^2) )
	
	# Draw gamma. 
	gamma_update <- rnorm(n= n_draws, mean = g_mean, sd= sqrt(g_var))
	return(gamma_update)
	}
	
muGibbsUpdate <- function(gamma_bar, tau, n_peptides)
	{
	# Determine the length of the arguments passed for vectorization.
	n_draws <- max(1, length(gamma_bar), length(tau), length(n_peptides))
	
	# Draw mu.
	mu_update <- rnorm(n= n_draws, mean= gamma_bar, sd= tau/sqrt(n_peptides))
	return(mu_update)
	}

sigmaGibbsUpdate <- function(n_states_total, sum_of_squares, alpha, beta)
	{
	# Determine the length of the arguments passed for vectorization.	
	n_draws <- max(1, length(n_states_total), length(sum_of_squares), length(alpha), length(beta))
	
	sigma_update <- sqrt(1/rgamma(n= n_draws, shape= alpha + n_states_total/2, rate= beta + sum_of_squares/2))
	return(sigma_update)
	}

tauMarginalGibbsUpdate <- function(n_peptides, gamma_bar, marginal_sum_of_squares, alpha, beta)
	{
	# Determine the length of the arguments passed for vectorization.	
	n_draws <- max(1, length(n_peptides), length(marginal_sum_of_squares), length(alpha), length(beta))
	
	tau_update <- sqrt(1/rgamma(n= n_draws, shape= as.vector(alpha - 1 + (n_peptides -1)/2), 
								rate= as.vector(beta + marginal_sum_of_squares/2) ))
	return(tau_update)
	}

tauGibbsUpdate <- function(n_peptides, sum_of_squares, alpha, beta)
	{
		# Determine the length of the arguments passed for vectorization.	
		n_draws <- max(1, length(n_peptides), length(sum_of_squares), length(alpha), length(beta))
		
		tau_update <- sqrt(1/rgamma(n= n_draws, shape= as.vector(alpha + (n_peptides)/2), 
						rate= as.vector(beta + sum_of_squares/2) ))
		return(tau_update)
	}
	