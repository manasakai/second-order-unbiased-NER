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

# model preliminaries --------------------------------------------------

# Dimensions
n = 5 # number of units in each area
m_list = [10, round(10 ** (3/2)), 100, round(10 ** (5/2)), 1000] # number of areas
p = 2 # dimension of covariate

# Set true parameter value
theta_true = np.loadtxt(dir + 'ThetaTrue' + str(theta_true_ind) + '.csv', delimiter=',') # True parameter value
beta_true = theta_true[:p]
tausq_true = theta_true[-2]
sigsq_true = theta_true[-1]

# Others
K = 10000 # number of iterations
varX = np.array([[4, 1], [1, 1]])
meanX = [1, 2]

# Generate the simulation dataset ==================================================

# Define a function to generate the simulation dataset
def gen_data(index):
    (k, m) = index
    data = np.zeros((n*m, p+1))
    for i in range(m):
        v_i = np.random.normal(0, np.sqrt(tausq_true))  # random effect for area i
        for j in range(n):
            # Generate data
            x_ij = np.random.multivariate_normal(meanX, varX) # Generate a covariate vector x~N(meanX, varX)
            epsilon_ij = np.random.normal(0, np.sqrt(sigsq_true))
            y_ij = x_ij@beta_true + v_i + epsilon_ij
            data[n*(i-1)+j, 1:] = x_ij
            data[n*(i-1)+j, 0] = y_ij
    return(data)

if __name__ == '__main__':

    # Generate data for each k=1,...,K --------------------------------------------------

    for m in m_list:
        indices = [(k, m) for k in range(K)]

        # Initialize a matrix to save data
        data = np.zeros((K, n*m, p+1))

        # Compute the matrix elements
        with mp.Pool(18) as pool:
            results = pool.map(gen_data, indices)
        for index, value in zip(indices, results):
            (k, m) = index
            data[k, :, :] = value

        # Save the results
        np.save(folder + 'data_m=' + str(m) + '.npy', data) # data is a  (K, m*n, p+1)-dimensional array
        # The last dimension corresponds to  (y_ij, x_ij)

    # # Plot --------------------------------------------------

    # k = 0
    # m = m_list[0]

    # data = np.load(folder + 'data_m=' + str(m) + '.npy')
    # print(data)

    # xbeta = data[k, :, 1:]@beta_true
    # y = data[k, :, 0]
    # print(np.corrcoef(xbeta, y))

    # import plotly.graph_objects as go
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Scatter(
    #         x=xbeta,
    #         y=y,
    #         mode='markers',
    #     )
    # )

    # # Show the figure
    # fig.show()
