# Preliminaries ==================================================

# Import libraries
import numpy as np
np.random.seed(0) # Set seed
import multiprocess as mp
import os

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

# Model preliminaries --------------------------------------------------

# Dimensions
n = 5 # number of units in each area
m_list = [10, round(10 ** (3/2)), 100, round(10 ** (5/2)), 1000] # number of areas

# Set true parameter value
theta_true = np.loadtxt(dir + 'ThetaTrue' + str(theta_true_ind) + '.csv', delimiter=',')

# Others
K = 10000 # number of iterations
warmup = 1000 # warm up period for MCMC
N = 20000 # length of each chain (after warm up period)

# Define the function to carry out the Gibbs sampling ==================================================

def NER_Gibbs_IG(input):
    (k, m) = input[:2]
    (a_tau, b_tau, a_sigma, b_sigma) = input[2:]

    # Data --------------------------------------------------

    data = np.load(folder + 'data_m=' + str(m) + '.npy')[k, :, :]
    y = data[:, 0] # y is an (n*m)-dimensional vector
    X = data[:, 1:] # X is an (n*m, p)-dimensional matrix
    XT = X.T.copy()
    XT_X_inv = np.linalg.inv(XT @ X)
    XT_X_inv_XT = XT_X_inv @ XT

    # Run the Gibbs sampler --------------------------------------------------

    # Shape parameters of inverse gamma distributions
    shape_siginv = n * m / 2 + a_sigma
    shape_tauinv = m / 2 + a_tau

    # Set initial values
    vstar_temp = np.zeros(m*n)
    tausq_temp = 2
    sigsq_temp = 2
    xi_temp = sigsq_temp + n * tausq_temp

    # Initialize the array to store samples
    theta_s = np.zeros((N, len(theta_true)))

    # Run the Gibbs sampler
    for t in range(warmup + N):

        # Sample beta from Gaussian distribution
        cov =  XT_X_inv / sigsq_temp
        mean = XT_X_inv_XT @ (y - vstar_temp)
        beta_temp = np.random.multivariate_normal(mean, cov)

        # Sample v = (v_1, ..., v_m) jointly from Gaussian distribution
        var = sigsq_temp * tausq_temp / xi_temp
        scale = np.sqrt(var)
        v_temp = np.zeros(m)
        for i in range(m):
            y_i = y[n*i : n*(i+1)]
            x_i = X[n*i : n*(i+1), :]
            mean = np.sum(y_i - x_i @ beta_temp) * tausq_temp / xi_temp
            v_temp[i] = np.random.normal(mean, scale)
        vstar_temp = np.kron(v_temp, np.ones(n))

        # Sample tausq_inv from Gamma distribution
        scale_denom = np.dot(v_temp, v_temp) + 2 * b_tau
        scale = 2 / scale_denom
        tausq_inv_temp = np.random.gamma(shape_tauinv, scale)
        tausq_temp = 1 / tausq_inv_temp

        # Sample sigsq_inv from Gamma distribution
        ep_temp = y - X @ beta_temp - vstar_temp
        scale_denom = np.dot(ep_temp, ep_temp) + 2 * b_sigma
        scale = 2 / scale_denom
        sigsq_inv_temp = np.random.gamma(shape_siginv, scale)
        sigsq_temp = 1 / sigsq_inv_temp

        # Update xi = sigsq + n * tausq
        xi_temp = sigsq_temp + n * tausq_temp

        # Save results (without the warmup period)
        if t >= warmup:
            theta_s[t - warmup, :] = np.hstack([beta_temp, tausq_temp, sigsq_temp])

    print('m =', m, 'completed for iteration', k)

    # Discard warmup samples
    return(theta_s)

# Carry out the Gibbs sampling ==================================================

# Set hyperparameters
a_tau = 5
b_tau = 5
a_sigma = 5
b_sigma = 5

# Gibbs sampling for each (k,m)
if __name__ == '__main__':

    # Initialize a matrix to save data
    sample_all = np.zeros((K,  len(m_list), N, len(theta_true)))

    for m in m_list:
        inputs = [(k, m, a_tau, b_tau, a_sigma, b_sigma) for k in range(K)] # (k, m) is the index

        with mp.Pool(18) as pool:
            results = pool.map(NER_Gibbs_IG, inputs)
        for (input, value) in zip(inputs, results):
            (k, m) = input[:2]
            sample_all[k, m_list.index(m), :, :] = value

    # Save the samples
    np.save(folder + 'MCMCsample_DG.npy', sample_all) # sample_all is a (K,  len(m_list), N, len(theta_true))-dimensional array
