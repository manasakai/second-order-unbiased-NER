# Preliminaries ==================================================

# Import libraries
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
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
m_list = [10, round(10**(3/2)), 100, round(10**(5/2)), 1000] # number of areas

# Set true parameter value
theta_true = np.loadtxt(dir + 'ThetaTrue' + str(theta_true_ind) + '.csv', delimiter=',') # True parameter value
theta_list = ['beta1', 'beta2', 'tausq', 'sigsq']
theta_names_list = ['$\\beta_{1}$', '$\\beta_{2}$', '$\\tau^{2}$', '$\\sigma^{2}$']

# List of prior distributions
prior_list = ['AU', 'JF', 'DG']

# Indices of theta, m, and prior
theta_indices = {theta: i for i, theta in enumerate(theta_list)}
m_indices = {m: i for i, m in enumerate(m_list)}
prior_indices = {prior: i for i, prior in enumerate(prior_list)}

# Plot preliminaries --------------------------------------------------

# List of plots
plot_list = ['Bias', 'MSE', 'CoverageProbability']

# Define color mapping
prior_colors = {
    'AU': 'dodgerblue',
    'JF':'tomato',
    'DG': 'mediumseagreen'
}

# Load csv files --------------------------------------------------

bias = np.zeros((len(m_list), len(prior_list), len(theta_list)))
MSE = np.zeros((len(m_list), len(prior_list), len(theta_list)))
coverage = np.zeros((len(m_list), len(prior_list), len(theta_list)))

for (prior, prior_index) in prior_indices.items():
    bias_prior = np.loadtxt(folder + 'bias_' + prior + '.csv', delimiter=',')
    MSE_prior = np.loadtxt(folder + 'MSE_' + prior + '.csv', delimiter=',')
    coverage_prior = np.loadtxt(folder + 'CoverageProbability_' + prior + '.csv', delimiter=',')

    bias[:, prior_index, :] = bias_prior
    MSE[:, prior_index, :] = MSE_prior
    coverage[:, prior_index, :] = coverage_prior

# Plots  =================================================

xaxis_title_text = '$\\log_{10}(m)$'

for plot in plot_list:
    # Plot setting
    if plot == 'Bias':
        yaxis_title_text = '$|E(\\hat{\\theta}_{r}-\\theta_{r})|$'
        title_standoff = 0
        margin = dict(l=70, r=10, t=40, b=10)
    if plot == 'MSE':
        yaxis_title_text = '$\\log_{10} E[(\\hat{\\theta}_{r}-\\theta_{r})^2]$'
        title_standoff = 0
        margin = dict(l=60, r=10, t=40, b=10)
    if plot == 'CoverageProbability':
        yaxis_title_text = 'Coverage probability'
        title_standoff = 10
        margin = dict(l=60, r=10, t=40, b=10)

    # Make sutplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=theta_names_list,  # Titles for each subplot
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.13,
        vertical_spacing=0.18
    )

    for (theta, theta_index) in theta_indices.items():
        row = (theta_index // 2) + 1  # Calculate row index (1-based)
        col = (theta_index % 2) + 1   # Calculate column index (1-based)

        for (prior, prior_index) in prior_indices.items():
            if plot == 'Bias':
                y = np.abs(bias[:, prior_index, theta_index])
            if plot == 'MSE':
                y = np.log10(MSE[:, prior_index, theta_index])
            if plot == 'CoverageProbability':
                y = coverage[:, prior_index, theta_index]

            fig.add_trace(
                go.Scatter(
                    x=np.log10(m_list),
                    y=y,
                    name=prior if theta_index == 0 else None,
                    mode='lines+markers',
                    opacity=1,
                    line=dict(
                        color=prior_colors[prior],
                        width=1
                    ),
                    marker=dict(size=5),
                    showlegend=(theta_index == 0)
                ),
                row=row,
                col=col
            )

            # Add base line for coverage probability
            if plot == 'CoverageProbability':
                fig.add_shape( # Horizontal line at y = 0.95
                    type='line',
                    x0=1,
                    x1=3,
                    y0=0.95,
                    y1=0.95,
                    opacity=1,
                    line=dict(
                        color='black',
                        dash='dash',
                        width=1
                    ),
                    row=row,
                    col=col
                )

        fig.update_xaxes(
            title_text=xaxis_title_text
        )
        fig.update_yaxes(
            title_text=yaxis_title_text,
            title_standoff=title_standoff
        )
    # Adjust the position of subplot_titles
    for annotation in fig['layout']['annotations']:
        annotation['yanchor'] = 'bottom'  # Place the annotation based on the bottom
        annotation['y'] -= 0.02  # Move the annotation closer to the graph

    # Update layout
    fig.update_layout(
        font=dict(family='Times New Roman'),
        margin=margin,
        legend=dict(
            title_text='Prior',
            x=1.02,# Adjust position
            y=1,
            traceorder='normal',
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        height=500,
        width= 800
    )

    # Save the figure
    fname = plot + '_summary'
    fig.write_image(folder+fname+'.svg')

    # Convert svg to pdf
    renderPDF.drawToFile(svg2rlg(folder+fname+'.svg'), folder+fname+'.pdf')

    # Remove the svg file
    os.remove(folder+fname+'.svg')
