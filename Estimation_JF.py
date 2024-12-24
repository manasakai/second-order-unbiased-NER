# Preliminaries ==================================================

# Import libraries
import numpy as np
np.random.seed(0) # Set seed
import multiprocess as mp
from scipy import sparse
from scipy.stats import gamma
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
warmup =100 # warm up period for MCMC
N = 2000 # length of each chain (after warm up period)

# iota denotes the n-dimensional vector of ones
iota = np.ones((n, 1))
iota_iota = iota @ iota.T

# Define the inverse of \bar{V} matrix
def Vbar_inv(rho, sigsq):
    return (1 / sigsq) * (np.eye(n) - ((1 - rho) / n) * iota_iota)

# Define a function that conducts sampling from the truncated gamma distribution truncated on (0,1), using inverse transform sampling
def truncated_gamma_rvs(shape, scale):
    gamma_dist = gamma(a = shape, scale = scale)
    F0 = gamma_dist.cdf(0)
    F1 = gamma_dist.cdf(1)
    c = F1 - F0
    U = np.random.uniform(0, 1, size=1)
    return  gamma_dist.ppf(F0 + c * U)

# Define the function to carry out the Gibbs sampling ==================================================

def NER_Gibbs(index):
    (k, m) = index

    # Data --------------------------------------------------

    data = np.load(folder + 'data_m=' + str(m) + '.npy')[k, :, :]
    y = data[:, 0] # y is an (n*m)-dimensional vector
    X = data[:, 1:] # X is an (n*m, p)-dimensional matrix
    XT = X.T.copy()

    # Run the Gibbs sampler --------------------------------------------------

    # Set initial values
    rho_temp = 0.5
    sigsq_temp = 2

    # Initialize the array to store samples
    theta_s = np.zeros((N, len(theta_true)))

    # Run the Gibbs sampler
    for t in range(warmup+N):

        # Sample beta from Gaussian distribution
        mat_temp = sparse.kron(np.eye(m), Vbar_inv(rho_temp, sigsq_temp), format='csr')
        cov =  np.linalg.inv(XT @ mat_temp @ X)
        mean = cov @ (y.T @ mat_temp @ X)
        beta_temp = np.random.multivariate_normal(mean, cov)

        # Sample rho from truncated gamma distribution using inverse transform sampling
        mat_temp = sparse.kron(np.eye(m), iota@iota.T, format='csr')
        res_temp = y - X @ beta_temp # residual
        scale_denom = res_temp.T @ mat_temp @ res_temp # denominator of the scale parameter
        scale = 2 * n * sigsq_temp / scale_denom # scale parameter
        shape = m / 2 # shape parameter
        rho_temp = truncated_gamma_rvs(shape, scale)

        # Sample sigsq_inv from Gamma distribution
        mat = np.eye(n, dtype=np.float64) - ((1 - rho_temp) / n) * iota_iota
        mat_temp = sparse.kron(np.eye(m), mat, format='csr')
        shape = n * m / 2 # shape parameter
        scale_denom = res_temp.T @ mat_temp @ res_temp # denominator of the scale parameter
        scale = 2 / scale_denom # scale parameter
        sigsq_inv_temp = np.random.gamma(shape, scale)
        sigsq_temp = 1 / sigsq_inv_temp

        # Update tausq by substituting tausq = sigsq * (1 - rho) / (n * rho)
        tausq_temp = sigsq_temp * (1 - rho_temp) / (n * rho_temp)

        # Save results (without the warmup period)
        if t >= warmup:
            theta_s[t - warmup, :] = np.hstack([beta_temp, tausq_temp, sigsq_temp])

    print('m =', m, 'completed for iteration', k)

    # Discard warmup samples
    return(theta_s)

# Carry out the Gibbs sampling ==================================================

# Gibbs sampling for each (k,m)
if __name__ == '__main__':

    # Initialize a matrix to save data
    sample_all = np.zeros((K,  len(m_list), N, len(theta_true)))

    for m in m_list:
        indices = [(k, m) for k in range(K)]

        with mp.Pool(18) as pool:
            results = pool.map(NER_Gibbs, indices)
        for (index, value) in zip(indices, results):
            (k, m) = index
            sample_all[k, m_list.index(m), :, :] = value

    # Save the samples
    np.save(folder + 'MCMCsample_JF.npy', sample_all) # sample_all is a (K,  len(m_list), N, len(theta_true))-dimensional array
