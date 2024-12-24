# This script contains functions that compute the inefficiency factor (IF) and the effective sample size (ESS) to check the convergence of MCMC.

# Preliminaries ==================================================

# Import libraries
import numpy as np
import plotly.graph_objects as go

# ====================================================================

# Define the autocovariance function
def acvf(x, lag): # x is a 1D array, lag is a natural number smaller than the length of x
    T = len(x)
    x_mean = np.mean(x)
    if T>lag:
        if lag == 0:
            return(np.sum((x - x_mean) ** 2) / T)
        else:
            return(np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / (T - lag))
    else:
        return('The lag is too large.')

# Define the autocorrelation function
def acf(x, lag): # x is a 1D array
    return acvf(x, lag) / acvf(x, 0)

# Define a function to plot ACFs
def ACFplot(x, max_t): # max_t is the maximum time lag that we want to see in the plot
    ACFs = np.zeros(max_t + 1) # initialize a vector to contain ACF for t=1,...,k.
    for t in range(max_t + 1):
        ACFs[t] = acf(x, t)

    # Plot ----------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
                x=np.arange(max_t+1),
                y=ACFs,
                marker_color='midnightblue',
                showlegend=False,
            ),
    )

    # Update layout
    fig.update_layout(
        title_text='ACF plot',
        xaxis_title_text='lag',
        yaxis_title_text='ACF',
        barmode='group',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        height=600,
    )

    # Show the figure
    fig.show()

# Define a function that returns the first k that satisfies acvf(k)<0.
def first_negative_acvf(x):
    T = len(x)
    t = 0
    while acvf(x, t) >= 0 and t < T:
        t = t + 1
    return(t)

# Define the inefficiency factor
def IF(x):
    T = len(x)
    k = first_negative_acvf(x)
    vec = np.zeros(k) # initialize a vector
    for t in range(k):
        vec[t] = (T - (t+1)) * acf(x, t+1)
    out = 1 + 2 * np.sum(vec) / T
    if out > 1:
        return out
    else:
        return 1

# Define the effective sample size
def ESS(x):
    T = len(x)
    return(T / IF(x))

# # Example ====================================================================

# T = 100
# x = np.zeros(T)
# x[0] = 1
# x[1] = 1.5

# # Define an AR(2) process
# for t in range(2, T):
#     x[t] = 0.7 * x[t-1] + 0.3 * x[t-2] + np.random.normal(0, 1)

# # Plot -------------------
# fig = go.Figure()

# fig.add_trace(
#     go.Scatter(
#         x=np.arange(1, T+1),
#         y=x,
#         mode='lines+markers'
#     )
# )

# # Update layout
# fig.update_layout(
#     xaxis_title_text='time',
#     height=600
# )

# # Show  the figure
# fig.show()

# print(first_negative_acvf(x))
# print('IF', IF(x))
# print('ESS', ESS(x))
# ACFplot(x, 99)