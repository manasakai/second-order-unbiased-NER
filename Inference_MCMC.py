# Preliminaries ==================================================

# Import libraries
import numpy as np
import multiprocess as mp
import os

# Import MCMC_convergence.py
import MCMC_convergence as mcmc

# Set true parameter index
theta_true_ind = 0 # 0 or 1

# Folder to save data, graphs, etc.
dir = os.path.dirname(__file__)
folder_name = 'output' + str(theta_true_ind)
folder = os.path.join(dir, folder_name)

# Choose OS (either Windows of Mac)
OS_list = ['win', 'mac']
OS = OS_list[0]
if OS == 'win':
    dir += '\\'
    folder += '\\'
if OS == 'mac':
    dir += '/'
    folder += '/'

# model preliminaries --------------------------------------------------

# Dimensions
n = 5 # number of units in each area
m_list = [10, round(10 ** (3/2)), 100, round(10 ** (5/2)), 1000] # number of areas

# Set true parameter value
theta_true = np.loadtxt(dir + 'ThetaTrue' + str(theta_true_ind) + '.csv', delimiter=',')
theta_list = ['beta1', 'beta2', 'tausq', 'sigsq']

# List of prior distributions
prior_list = ['AU', 'JF', 'DG']

# Indices of theta, m, and prior
theta_indices = {theta: i for i, theta in enumerate(theta_list)}
m_indices = {m: i for i, m in enumerate(m_list)}
prior_indices = {prior: i for i, prior in enumerate(prior_list)}

# Others
K = 10000 # number of simulation datasets

# Posterior inference =================================================
for (prior,prior_index) in prior_indices.items():

    # Load MCMC sample
    MCMCsample = np.load(folder + 'MCMCsample_' + prior +'.npy')

    # length of each chain (after warm up period)
    if prior == 'DG':
        N = 20000
    else:
        N = 2000

    #  Check MCMC convergence --------------------------------------------------

    # ACF, ESS, IF when k=k

    def IF_ESS(input):
        (k, m_index, theta_index) = input
        x = MCMCsample[k, m_index, :, theta_index]
        return(mcmc.IF(x), mcmc.ESS(x))

    if __name__ == '__main__':

        # Initialize a matrix to save data
        IF_mat = np.zeros((K,  len(m_list), len( theta_list)))
        ESS_mat = np.zeros((K,  len(m_list), len( theta_list)))

        for m_index in range(len(m_list)):
            for theta_index in range(len(theta_list)):
                inputs = [(k, m_index, theta_index) for k in range(K)]

                with mp.Pool(4) as pool:
                    results = pool.map(IF_ESS, inputs)
                for (input, value) in zip(inputs, results):
                    (k, m_index, theta_index) = input[:3]
                    IF_mat[k,  m_index, theta_index] = value[0]
                    ESS_mat[k,  m_index, theta_index] = value[1]

    minESS_index = np.unravel_index(np.argmin(ESS_mat), ESS_mat.shape) # Index of argmin ESS
    print('For prior ', prior, ', ESS is minimized at k=', range(K)[minESS_index[0]], ', m=', m_list[minESS_index[1]], ', parameter=', theta_list[minESS_index[2]], 'and ESS=', np.min(ESS_mat))

    maxIF_index = np.unravel_index(np.argmax(IF_mat), IF_mat.shape) # Index of argmax IF
    print('For prior ', prior, ', IF is maximized at k=', range(K)[maxIF_index[0]], ', m=', m_list[maxIF_index[1]], ', parameter=', theta_list[maxIF_index[2]], 'and IF=', np.max(IF_mat))

    # Compute the posterior mean --------------------------------------------------

    theta_hat = np.zeros((K, len(m_list), len(theta_list))) # posterior mean
    for k in range(K):
        for (m, m_index) in m_indices.items():
            for (theta, theta_index) in theta_indices.items():
                theta_hat[k, m_index,  theta_index] = \
                    np.mean(np.mean(MCMCsample[k, m_index, :, theta_index]))

    np.save(folder + 'theta_hat' + prior + '.npy', theta_hat) # theta_hat is a (K,  len(m_list), len(theta_list))-dimensional array

    # Compute mean, bias and MSE of Bayes estimator --------------------------------------------

    # Compute E[theta_hat], E[theta_hat - theta], and E[(theta_hat - theta)^2] for each m
    theta_hat_mean = np.mean(theta_hat, axis=0)
    theta_hat_bias = theta_hat_mean - theta_true
    theta_hat_MSE = np.mean((theta_hat - theta_true)**2, axis=0)

    # Save as csv
    np.savetxt(folder + 'mean_' + prior + '.csv', theta_hat_mean, delimiter=',')
    np.savetxt(folder + 'bias_' + prior + '.csv', theta_hat_bias, delimiter=',')
    np.savetxt(folder + 'MSE_' + prior + '.csv', theta_hat_MSE, delimiter=',')

    # Compute coverage probability --------------------------------------------------

    # Set alpha
    alpha = 0.95

    # Calculate the lower and upper quantiles
    lower_quantile = (1 - alpha) / 2
    upper_quantile = 1 - lower_quantile

    # Compute the coverage probability for each parameter
    coverage_probability = np.zeros((len(m_list), len(theta_list)))
    for (m, m_index) in m_indices.items():
        for (theta, theta_index) in theta_indices.items():
            theta0 = theta_true[theta_index]
            theta_samples = MCMCsample[:, m_index, :, theta_index] # (K, N)-dimensional
            lower = np.quantile(theta_samples, lower_quantile, axis=1) # Compute lower quantile (K-dimensional)
            upper = np.quantile(theta_samples, upper_quantile, axis = 1) # Compute upper quantile (K-dimensional)
            coverage = (theta0 >= lower) & (theta0 <= upper) # K-dimensional
            coverage_probability[m_index, theta_index] = np.mean(coverage)

    # Save as csv
    np.savetxt(folder + 'CoverageProbability_' + prior + '.csv', coverage_probability, delimiter=',')
