#!/usr/bin/env Rscript
# Individual state model with no random censoring and common variance.

# Note: Make executable by running: chmod ug+x [file] on this file.
rm(list= ls())
# Local library location on Odyssey.
local_library <- '/n/airoldifs1/ejsolis/R/' 
.libPaths(local_library)

options(width= 120)
require('ggplot2')
require('plyr')
require('Matrix')
require('mvtnorm')
require('coda')
tryCatch(expr= require('stringr'), warning= function(x) library('stringr', lib.loc = local_library))
tryCatch(expr= require('speedglm'), warning= function(x) library('speedglm', lib.loc = local_library))
tryCatch(expr= require('getopt'), warning= function(x) library('getopt', lib.loc = local_library))
tryCatch(expr= require('optparse'), warning= function(x) library('optparse', lib.loc = local_library))

RNGkind(normal.kind = "Ahrens-Dieter")

# Parse the command line options.
option_list <- list(
		make_option(c("-f", "--functions_file"), action= 'store', 
				type= 'character', 
				help= 'File with functions used in the fitting.'),
		make_option(c("-v", "--verbose"), action="store_true", default=TRUE,
				help="Print extra output [default]"),
		make_option(c("-q", "--quant_file"), action="store", type= "character",
				dest="quant_file", help="Protein quantitation file."),
		make_option(c("-o", "--out_path"), type="character",
				help="Path where the results folder should be created.",
				dest="out_path"),
		make_option(c("--experiment"), type= 'integer', dest= 'experiment',
				help = "Experiment number to quantitate proteins in."),
		make_option(c("-n", "--n_iter"), default=1000, type= 'integer',
				dest= 'n_iter', help="Number of  iterations [default %default]"),
		make_option(c('-t', '--temp_save'), default=300, dest="temp_save_freq",
				help="Frequency to save simulations. [default %default]")
	)
# get command line options, if help option encountered print help and exit,
# otherwise if options not found on command line then set defaults,
options <- parse_args(OptionParser(option_list=option_list))

print(options)

# Individual state model with no random censoring and common variance.

# Read in functions.
source(file= options$functions_file)

#--- Read in quantitation data. ---#
#experiments <- c('ups2_3h_1', 'ups2_3h_2', 'ups2_6h_1', 'ups2_90m_1', 'ups2_90m_2')
experiments <- c("ups2_mdy748_8h_1", "ups2_mdy748_8h_2", "ups2_mdy748_8h_3")
restrict_to_experiments <- experiments[options$experiment]

quant_file <- options$quant_file

quant_data	<- read.table(	file= quant_file, header= TRUE, sep= '\t', 
		stringsAsFactors= FALSE)


# Create file variables.
date_time <- str_replace_all(str_replace_all(Sys.time(), ' ', '_'), ':', '-')
out_path <- paste(options$out_path, '/', restrict_to_experiments, '_', date_time, sep= '')
dir.create(out_path)

temp_plts_file <- paste(out_path,'/Temp_Plots.pdf', sep= '')
print(temp_plts_file)
temp_model_fits_file <- paste( out_path,'/Temp_Fits', '.RData', sep= '')
print(temp_model_fits_file)
model_fits_file <- paste( out_path,'/Model_Fits', '.RData', sep= '')
print(model_fits_file)
workspace_file <- paste( out_path,'/Workspace', '.RData', sep= '')
print(workspace_file)
# Restrict to the specified experiments.
quant_data <- quant_data[quant_data$Experiment %in% restrict_to_experiments,]

# Use the leading razor protein from MaxQuant as the protein for each peptide.
quant_data$Protein <- quant_data$LeadingRazorProtein


# Create and observed indicator.
quant_data$Observed <- (quant_data$Intensity != 0)
quant_data[quant_data$Observed, 'Intensity'] <- log10(quant_data[quant_data$Observed, 'Intensity'])

quant_data$W <- ifelse(test= quant_data$Observed, yes= 0, no= NA)
# Remove unnecessary columns from quant data.
quant_data$State <- ifelse(	test= quant_data$Observed == TRUE,
		yes = 'Observed_State',
		no	= 'Censored_State'
)
quant_data <- quant_data[,c("Protein", "Peptide", "State", "Observed", "Intensity", "W")]


#--- Fit individual state model. ---#
n_iter <- options$n_iter

observed_peptides_df <- quant_data[quant_data$Observed,]


#X_intensity_obs <- 	sparse.model.matrix(Intensity ~ 0 + Protein, sparse= TRUE, 
#		data= observed_peptides_df)
#X_peptide_obs	<-	sparse.model.matrix(~0 + Peptide, sparse= TRUE, 
#		data= observed_peptides_df)
observed_intensity_df_2 <-	ddply( .data= quant_data,
		.(Peptide), transform, 
		N_Observed= sum(Observed))

peptide_data	<-	ddply(.data= observed_intensity_df_2, .(Peptide), summarise,
		Protein= Protein[1], N_Observed= sum(Observed))

## Get the proportion of peptides for each protein with at least on observed 
## state.
#prop_obs_peps_by_prot <- ddply(.data= peptide_data, .(Protein), summarise,
#								P_Pep_Obs= mean(N_Observed > 0))

n_peptides_total<-	length(table(peptide_data$Peptide))
protein_names <- names(table(peptide_data$Protein))

n_states_observed_by_protein <- ddply(.data= peptide_data, .(Protein), summarise,
		N_Observed= sum(N_Observed))
observed_prots <- n_states_observed_by_protein[which(n_states_observed_by_protein$N_Observed > 0), 'Protein'] 

# Restrict to proteins that have some identified peptides, ie weren't completely
# censored.
quant_data		<- quant_data[which(quant_data$Protein %in% observed_prots),]
peptide_data	<- peptide_data[which(peptide_data$Protein %in% observed_prots),]
observed_peptides_df <- observed_peptides_df[which(observed_peptides_df$Protein %in% observed_prots),]

observed_peps <- peptide_data[which(peptide_data$Protein %in% observed_prots), 'Peptide']

# Map peptides to proteins.
peptide_to_protein_map <- data.frame(Protein= as.character(peptide_data$Protein))
rownames(peptide_to_protein_map) <- peptide_data$Peptide

peptide_indicators <- sparse.model.matrix(~ 0 + Peptide, data= peptide_data)
protein_indicators <- sparse.model.matrix(~ 0 + Protein, data= peptide_data)


#--- Parameter storage. ---#
mu				<- matrix(nrow= n_iter, ncol= length(observed_prots))
colnames(mu)	<- protein_names[which(protein_names %in% observed_prots)] 
tau				<- matrix(nrow= n_iter, ncol= length(observed_prots))
colnames(tau)	<- protein_names[which(protein_names %in% observed_prots)]

# Peptide level parameters.
gamma			<- matrix(nrow= n_iter, ncol= length(observed_peps))
colnames(gamma) <- observed_peps
sigma			<- matrix(nrow= n_iter, ncol= length(observed_prots))
colnames(sigma) <- protein_names[which(protein_names %in% observed_prots)]

# Censoring parameters.
eta			<- matrix(nrow= n_iter, ncol= 2)
p_rc		<- matrix(nrow= n_iter, ncol= 1)
n_cen		<- matrix(nrow= n_iter, ncol= length(peptide_data$Peptide))
n_cen_total	<- matrix(nrow= n_iter, ncol= 1)
colnames(n_cen) <- peptide_data$Peptide

# Hyper parameters.
lambda		<- matrix(nrow= n_iter, ncol= 1)
r			<- matrix(nrow= n_iter, ncol= 1)
alpha_sigma <- matrix(nrow= n_iter, ncol= 1)
beta_sigma	<- matrix(nrow= n_iter, ncol= 1)
alpha_tau	<- matrix(nrow= n_iter, ncol= 1)
beta_tau	<- matrix(nrow= n_iter, ncol= 1)

# Save summed unlogged intensity by protein.
SummedUnloggedIntensity <- matrix(nrow= n_iter, ncol= length(observed_prots))
colnames(SummedUnloggedIntensity)	<- protein_names[which(protein_names %in% observed_prots)]

#--- Initialize parameters. ---#

# Initialize hyper parameters at a reasonable constant.

#lambda[1]	<-	(1/mean(peptide_data$N_Observed))
#lambda[1]   <-  .9
#r[1]		<-	1

# Compute the MLEs of the negative binomial parameters using the number of 
# observed states for peptides with at least one observed state.
lambda_mu_initial_mles <- computeNbinomMLEs(peptide_data[peptide_data$N_Observed > 0, 'N_Observed'])
lambda[1] 	<- 1/(1+exp(-lambda_mu_initial_mles$logitLambdaMLE))
r[1]		<- exp(lambda_mu_initial_mles$logRMLE)
print(paste('Lambda and r initialized at MLEs with values:',lambda[1], 'and', r[1]))

# Initialize the variance hyper parameters.
alpha_sigma[1] 	<- 4
beta_sigma[1]	<- 2
hist(sqrt(1/rgamma(n= 1000, shape= alpha_sigma[1], rate= beta_sigma[1])))

alpha_tau[1]	<- 4
beta_tau[1]		<- 2
hist(sqrt(1/rgamma(n= 1000, shape= alpha_tau[1], rate= beta_tau[1])))


# Initialize the protein level parameters based on observed peptides.
#obs_prot_means 	<- tapply(	X		= observed_peptides_df$Intensity,
#							INDEX	= observed_peptides_df$Protein,
#							FUN		= mean)
#prot_means 	<- tapply(		X		= quant_data$Intensity,
#							INDEX	= quant_data$Protein,
#							FUN		= mean)
#					
#pep_means		<- tapply(	X		= quant_data$Intensity,
#							INDEX	= quant_data$Peptide,
#							FUN		= mean)
#
#obs_pep_means	<- tapply(	X		= observed_peptides_df$Intensity,
#							INDEX	= observed_peptides_df$Peptide,
#							FUN		= mean)
#					
#obs_prot_s_squared <- sapply(names(obs_prot_means), 
#					FUN= function(x)  sum( ( obs_pep_means[which(peptide_to_protein_map[names(obs_pep_means),] == x)] -
#											- obs_prot_means[x] )^2) / 
#				(length(obs_pep_means[which(peptide_to_protein_map[names(obs_pep_means),] == x)])))
#
#sigma[1,]		<- tapply(X= observed_peptides_df$Intensity, INDEX= observed_peptides_df$Protein, FUN= sd)
#sigma[1,which(is.na(sigma[1,]))] <- mean(sigma[1,], na.rm= TRUE)
#
#
#obs_pep_s_squared <- sapply(names(obs_prot_means), 		
#					FUN= function(x)  sum( ( observed_peptides_df[which(observed_peptides_df$Protein == x), 'Intensity'] -
#					- obs_pep_means[observed_peptides_df[which(observed_peptides_df$Protein == x),'Peptide']] )^2) / 
#					(length(observed_peptides_df[which(observed_peptides_df$Protein == x), 'Intensity'])) )

obs_pep_stats_df <- 	ddply(observed_peptides_df, .(Peptide, Protein), transform, 
		Pep_Mean= mean(Intensity))
obs_pep_stats_df <- 	ddply(obs_pep_stats_df, .(Protein), transform, 
		Prot_Mean= mean(Pep_Mean))
prot_means		 <-		ddply(obs_pep_stats_df, .(Protein), summarise,
		Prot_Mean= mean(Pep_Mean))
sigma_initial_df <-		ddply(obs_pep_stats_df, .(Protein), summarise,
		Pep_S= sqrt(mean( (Intensity-Pep_Mean)^2 )))
sigma_initial_df[which(sigma_initial_df$Pep_S == 0),"Pep_S"] <- mean(sigma_initial_df$Pep_S, na.rm= TRUE)
tau_initial_df <-		ddply(obs_pep_stats_df, .(Protein), summarise,
		Prot_S= sqrt(mean( (Pep_Mean-Prot_Mean)^2 )))
tau_initial_df[which(tau_initial_df$Prot_S == 0),"Prot_S"] <- mean(tau_initial_df$Prot_S, na.rm= TRUE)



sigma[1,sigma_initial_df$Protein] 	<- sigma_initial_df$Pep_S
tau[1,tau_initial_df$Protein]		<- tau_initial_df$Prot_S

# Initialize mu at 1 sigma below the observed mean.
mu[1,prot_means$Protein] <- prot_means$Prot_Mean- sigma[1,prot_means$Protein] 
gamma[1,obs_pep_stats_df$Peptide] <- obs_pep_stats_df$Pep_Mean - 
		sigma[1,peptide_to_protein_map[obs_pep_stats_df$Peptide,'Protein']] 
# Initialize unobserved peptides at 2 sigma below the observed protein mean.
gamma[1,which(is.na(gamma[1,]))] <- mu[1,peptide_to_protein_map[names(gamma[1,which(is.na(gamma[1,]))]),'Protein']]- 
		2 * sigma[1,peptide_to_protein_map[names(gamma[1,which(is.na(gamma[1,]))]),'Protein']]

#mu[1,]			<- prot_means#obs_prot_means 
#gamma[1,]		<- 	( (mu[1,peptide_data$Protein]/tau[1,peptide_data$Protein]^2) + 
#					(pep_means/(sigma[1,peptide_data$Protein]^2/peptide_data$N_Observed) ) ) / 
#					( (1/tau[1,peptide_data$Protein]^2) + (peptide_data$N_Observed/sigma[1,peptide_data$Protein]^2) )
##gamma[1,]		<- prot_means[peptide_data$Protein]


# Initialize the intensity-censoring GLM parameters at a reasonable value.
#all_peptides_glm <- speedglm(data= msms_data,formula= Observed ~ Intensity, 
#		family= binomial(link= 'logit'), sparse= FALSE)
#eta[1,] <- coef(all_peptides_glm)
#eta_hat		<- coef(all_peptides_glm)
#eta_hat_cov	<- vcov(all_peptides_glm)

#eta[1,] <- c(-12, 2.5)
eta[1,] <- c(-12, 4)

n_cen[1,] <- 0


# Draw lots of random normals.
rand_norms 		<- rnorm(n= 10000)


# Initialize the random censoring probability at the mean of the proportion of
# unobserved peptides.
p_rc[1,] <- .1#1- mean(peptide_data$N_Observed > 0)



## MCMC...
for(t in 2:n_iter)
{
	print(paste("Iteration:", t))
	print(paste("Experiment:", restrict_to_experiments))
#t = 2	
	
	# 1) Draw censored peptide intensities.
	# a) Characterize censored peptide intensity distributions.
	censored_intensity_stats_df <- characterizeCensoredIntensityDistribution(
			eta_0	= eta[t-1,1],
			eta_1	= eta[t-1,2], 
			mu		= gamma[t-1,peptide_data$Peptide],
			sigma	= sigma[t-1,peptide_data$Protein],
			TOL		= 1e-7,
			protein = peptide_data$Protein,
			peptide	= peptide_data$Peptide,
			MAX_ITER= 100,
			rand_normals = rand_norms)
	
#	censored_intensity_stats_df <- data.frame(
#			protein		= censored_intensity_stats$protein,
#			peptide		= censored_intensity_stats$peptide,
#			y_hat		= censored_intensity_stats$y_hat,
#			p_int_cen	= censored_intensity_stats$p_int_cen,
#			approx_sd	= censored_intensity_stats$approx_sd)
	
	
	
	# b) Draw the number of censored peptide states.
	# Use Block's rejection sampler.
	n_censored <- rnCen( 
			nobs	= peptide_data[which(peptide_data$Peptide == censored_intensity_stats_df$peptide),'N_Observed'],
			prc		= p_rc[t-1],
			pic		= censored_intensity_stats_df$p_int_cen,
			lmbda	= lambda[t-1],
			r		= r[t-1])
	n_by_pep 	<- n_censored + peptide_data[which(peptide_data$Peptide == censored_intensity_stats_df$peptide),'N_Observed']#peptide_data$N_Observed
	n_cen[t,] 	<- 	n_censored
	
	print(paste('Sum n_censored:', sum(n_censored)))
	n_cen_total[t] <- sum(n_censored)
	
	# c) Draw the censored intensities.
	cen_ints <- drawCensoredIntensitiesFast(
			n			= n_censored,
			mu			= gamma[t-1, censored_intensity_stats_df$peptide],
			sigma		= sigma[t-1, censored_intensity_stats_df$protein],
			y_hat		= censored_intensity_stats_df$y_hat,
			approx_sd	= censored_intensity_stats_df$approx_sd,
			p_ic		= censored_intensity_stats_df$p_int_cen,
			p_rc		= p_rc[t-1],
			eta_0		= eta[t-1,1],
			eta_1		= eta[t-1,2],
			df			= 1,
			protein		= as.character(censored_intensity_stats_df$protein),
			peptide		= as.character(censored_intensity_stats_df$peptide))
	
	
	# Combine the censored and observed data frames. 
	all_peptides_df	<- rbind(observed_peptides_df, cen_ints)
	n_peptides_by_prot <- tapply(	
			X		= all_peptides_df$Intensity,
			INDEX	= all_peptides_df$Protein,
			FUN		= length)
	
	# Save the total unlogged intensity by protein.
	SummedUnloggedIntensity[t,] <- tapply(	
			X		= all_peptides_df$Intensity,
			INDEX	= all_peptides_df$Protein,
			FUN		= function(x) sum(10^x)) 
	
	# 2) Draw the random censoring probability.
	n_rand_censored_total <- sum(all_peptides_df$W)
	p_rc[t] <- rbeta(n=1, shape1= n_rand_censored_total + 1,
			shape2= nrow(all_peptides_df) - n_rand_censored_total + 1)
	
	
	# 3) Update gamma, the peptide-level means.
	peptide_means <- tapply(X		= all_peptides_df$Intensity, 
			INDEX	= all_peptides_df$Peptide,
			FUN		= mean)
	
	
	gamma[t,]	<- gammaGibbsUpdate(
			mu		= as.vector(protein_indicators %*% mu[t-1,]),
			tau 	= as.vector(protein_indicators %*% tau[t-1,]),
			sigma	= as.vector(protein_indicators %*% sigma[t-1,]),
			y_bar	= as.vector(peptide_means),
			n_states= as.vector(n_by_pep))
	
	
	# 5) Draw mu, the protein-level means.
	gamma_bar 	<- t(protein_indicators) %*%matrix(data= gamma[t,], ncol= 1)/colSums(protein_indicators)
#	print('gamma bar:')
#	print(gamma_bar)
	peptide_n	<- colSums(protein_indicators)
	names(peptide_n) <- colnames(protein_indicators)
	mu[t,]	<- muGibbsUpdate(
			gamma_bar	= as.vector(gamma_bar),
			tau			= as.vector(tau[t-1,]),
			n_peptides	= as.vector(peptide_n))
	
	# 4) Draw tau, the peptide level variances.
	
	marginal_ss <- t(protein_indicators) %*% (gamma[t,] - (protein_indicators %*% gamma_bar))^2	
#	tau[t,]	<- tauMarginalGibbsUpdate(	n_peptides				= peptide_n,
#										gamma_bar				= gamma_bar,
#										marginal_sum_of_squares	= marginal_ss,
#										alpha					= alpha_tau[t-1],
#										beta					= beta_tau[t-1])	
	gamma_ss <- t(protein_indicators) %*% (gamma[t,] - (protein_indicators %*% mu[t,]))^2
	tau[t,]		<- tauGibbsUpdate(
			n_peptides		= peptide_n,
			sum_of_squares	= gamma_ss,
			alpha			= alpha_tau[t-1],
			beta			= beta_tau[t-1])
	print(paste('Min, Max Tau:', min(tau[t,]), max(tau[t,])))
	
	# 6) Update sigma, the state-level variances.
	int_m_gamma_ss <- sapply(colnames(mu), FUN = function(x) 
				sum( (all_peptides_df[which(all_peptides_df$Protein == x), 'Intensity'] -
									gamma[t,all_peptides_df[which(all_peptides_df$Protein == x), 'Peptide']])^2)) 
	
	
	sigma[t,]	<- sigmaGibbsUpdate(
			n_states_total	= as.vector(n_peptides_by_prot),
			sum_of_squares	= as.vector(int_m_gamma_ss),
			alpha			= alpha_sigma[t-1],
			beta			= beta_sigma[t-1]
	)
	print(paste('Min, Max Sigma:', min(sigma[t,]), max(sigma[t,])))
	
	# 7) Update alpha_sigma and beta_sigma, the state-level variances hyper 
	# parameters.
	print('sigma hyper parameters')
	a_b_sigma_updates	<- varianceHyperparametersMHUpdate(
			variances	= sigma[t,]^2,
			alpha_tm1	= alpha_sigma[t-1],
			beta_tm1	= beta_sigma[t-1])
	alpha_sigma[t]		<- a_b_sigma_updates[1]
	beta_sigma[t]		<- a_b_sigma_updates[2]	
#alpha_sigma[t]		<- alpha_sigma[1]
#beta_sigma[t]		<- beta_sigma[1]	
	
	
	print("alpha_sigma[t], beta_sigma[t]:")
	print(c(alpha_sigma[t], beta_sigma[t]))
	
	# 8) Update alpha_tau and beta_tau, the peptide-level variance hyper 
	# parameters.
	print('tau hyper parameters')
	a_b_tau_updates	<- varianceHyperparametersMHUpdate(
			variances	= tau[t,]^2,
			alpha_tm1	= alpha_tau[t-1],
			beta_tm1	= beta_tau[t-1])
	alpha_tau[t]		<- a_b_tau_updates[1]
	beta_tau[t]			<- a_b_tau_updates[2]
#	alpha_tau[t]		<- alpha_tau[1]
#	beta_tau[t]			<- beta_tau[1]
	
	
	
	print("alpha_tau[t], beta_tau[t]:")
	print(c(alpha_tau[t], beta_tau[t]))
	
	
	
	# 6) Update Eta.
	all_data_df <- all_peptides_df[which(all_peptides_df$W == 0),]
#	all_data_df <- msms_data	
	
	
	eta_model 	<- speedglm(data= all_data_df,formula= Observed ~ Intensity, 
			family= binomial(link= 'logit'), sparse= FALSE, 
			eigendec= FALSE)
	eta_hat		<- coef(eta_model)
	eta_hat_cov	<- vcov(eta_model)
	eta_star	<- as.numeric(rmvnorm(n= 1, mean= eta_hat, sigma= eta_hat_cov))
	# Compute the prior and importance ratios.
	p_eta_star	<- 1/(1 + exp(-1*(eta_star[1] + eta_star[2] * 
									all_data_df[,'Intensity'])))
	p_eta_prev	<- 1/(1 + exp(-1*(eta[t-1,1] + eta[t-1,2] *
									all_data_df[,'Intensity'])))
	lpr_eta 	<- sum(
			dbinom(all_data_df[,'Observed'],1,p_eta_star, log= TRUE) - 
					dbinom(all_data_df[,'Observed'],1,p_eta_prev, log= TRUE)
	) 
	lir_eta 	<- dmvnorm(
					x= eta_star, mean= eta_hat,sigma= eta_hat_cov,log= TRUE)-
			dmvnorm(x= eta[t-1,], mean= eta_hat, sigma= eta_hat_cov, log= TRUE)
	log_r_eta	<- lpr_eta - lir_eta
	log_u_eta	<- log(runif(n= 1))
	# MH step.
	mh_update_eta <- log_u_eta < log_r_eta
	if(mh_update_eta == TRUE)
	{
		eta[t,] <- eta_star
	}else{	
		eta[t,] <- eta[t-1,]
	}
	print(paste('Eta MH Update:', mh_update_eta))
	
	# 7) Draw the negative binomial size parameter.
	
	# Random walk Metropolis for r.
	
	# Second joint r, lambda MH.
	r_lambda_updates <- rLambdaMH2(	n_by_pep= n_by_pep, lambda_tm1= lambda[t-1],
			r_tm1= r[t-1], r_prior= c(.25, .25/2))#c(3,3/2))#c(200, 200/.75))
	lambda[t] 	<- r_lambda_updates[1]
	r[t] 		<- r_lambda_updates[2]
	
	
	
	
#	print(censored_intensity_stats_df)
	print('mu[t,]:')
	print(mu[t,])
	print(paste('sigma[t]:', sigma[t]))
	print(paste("lambda[t]:", lambda[t]))
	print('eta[t,]:')
	print(eta[t,])
	print(paste('p_rc[t]:', p_rc[t]))
	print(paste('r[t]:', r[t]))
#	print(paste('r_tilde:',r_tilde))
#	print(paste('r_star:',r_star))
	print('')
	
	# Quit if the total number of censored peptides is too large.
	if(n_cen_total[t] > 1000000)
	{
		break()
	}
	
	
	if(t %% options$temp_save_freq == 0)
	{		
		# Save the temporary results.
		temp_results <-  mcmc(data.frame(eta= eta, lambda= lambda, r= r, 
						p_rc= p_rc, alpha_sigma= alpha_sigma,  
						beta_sigma= beta_sigma, alpha_tau= alpha_tau, 
						beta_tau= beta_tau, n_cen_total= n_cen_total, mu= mu, 
						sigma= sigma, n_cen= n_cen, gamma= gamma, tau= tau,
						summed_ul_int= SummedUnloggedIntensity)) 
		temp_results <- na.omit(temp_results)				
		save(temp_results, file= temp_model_fits_file)		
		
	}
	
# End main mcmc loop. 
}


# Save the results.
results <- mcmc(data.frame(eta= eta, lambda= lambda, r= r, p_rc= p_rc, 
				alpha_sigma= alpha_sigma, beta_sigma= beta_sigma, 
				alpha_tau= alpha_tau, beta_tau= beta_tau, 
				n_cen_total= n_cen_total, mu= mu, sigma= sigma, 
				n_cen= n_cen, gamma= gamma, tau= tau,
				summed_ul_int= SummedUnloggedIntensity))

#results <- na.omit(results)
#results <- tail(results, nrow(results) * .75)


# Save the fits.

save(results, file= model_fits_file)

# Save workspace.
save.image(workspace_file, compress = 'xz')

