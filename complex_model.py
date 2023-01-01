# complex models

# libraries

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Vasicek

# r_0: initial interest rate
# k: speed of mean reversion
# theta: target interest rate
# sig: sigma term
# t: time
# dt: time length


def vasicek(r_0, k, theta, sig, t, dt):

    # Organizing vasicek model

    # DataFrame to hold interest rate tree

    model_df = pd.DataFrame(index=range(0, t), columns=range(0, t))

    # Organizing trees

    model_df[0][0] = r_0

    for k_col in range(1, t):
        for j in range(0, t):
            if model_df.columns[k_col] < model_df.index[j]:
                pass

            # Tree with risk-neutral probability of 1/2
            elif k_col % 2 != 0:
                if j == np.trunc(k_col / 2):
                    model_df[k_col][j] = model_df[k_col-1][j] + k * \
                        (theta - model_df[k_col-1][j]) * dt + sig * np.sqrt(dt)
                elif j == np.ceil(k_col / 2):
                    model_df[k_col][j] = model_df[k_col][j-1] - \
                        2 * sig * np.sqrt(dt)

                    # Tree to recombine
                    try:
                        temp_average = (
                            model_df[k_col][j-1] + model_df[k_col][j]) / 2
                        model_df[k_col+1][j] = temp_average + \
                            k * (theta - temp_average) * dt

                    # To ignore error caused by t
                    except KeyError:
                        pass
            else:
                pass

    # Tree with p, q, r_u, r_d
    while math.isnan(model_df[t-1][0]) == True:

        for k_col in range(1, t):
            for j in range(0, t):

                if math.isnan(model_df[k_col][j]) == True and (k_col / 2) > j:

                    temp_expectation = model_df[k_col-1][j] + \
                        k * (theta - model_df[k_col-1][j]) * dt
                    model_df[k_col][j] = temp_expectation + \
                        (sig ** 2 * dt) / \
                        (temp_expectation - model_df[k_col][j+1])

                elif math.isnan(model_df[k_col][j]) == True and (k_col / 2) < j:

                    temp_expectation = model_df[k_col-1][j-1] + \
                        k * (theta - model_df[k_col-1][j-1]) * dt
                    model_df[k_col][j] = temp_expectation - \
                        (sig ** 2 * dt) / \
                        (model_df[k_col][j-1] - temp_expectation)

    # For visualization (return to np.float64 for aftermath / Search for a better code required)
    model_df = model_df.astype(float).round(4)

    # Plotting

    # Will be used for plotting binomial tree with matplotlib.pyplot
    plot_df = model_df.copy()
    plot_df = plot_df.loc[::-1]

    for i in range(t):
        plot_df[i] = plot_df[i].shift(i - t + 1)

    # Basic settings
    fig = plt.figure(figsize=(25, 25))
    plt.xlabel('time index')
    plt.ylabel('interest rate')
    plt.title('Vasicek model')
    plt.xlim(-1, t + 1)

    for i in range(t):
        plt.plot(model_df.columns,
                 model_df.iloc[i],
                 plot_df.columns,
                 plot_df.iloc[i],
                 linewidth=0.5,
                 marker='o',
                 ms=7,
                 markeredgecolor='white',
                 markeredgewidth=3,
                 color='#282c2e')

    # Settings for NaN cells & reveal values
    for i in range(t):
        for j, k in zip(model_df.columns, model_df.iloc[i]):
            if math.isnan(k) == True:
                pass
            else:
                plt.text(j,
                         k,
                         str(k),
                         color='blue',
                         fontsize=10,
                         horizontalalignment='right',
                         verticalalignment='top')

    # Plotting model in VSCode (Different from Jupyter notebook)
    plt.show()

    # Dataframe styling

    # Matching timestamps
    model_df.columns = model_df.columns * dt
    model_df.index = model_df.index * dt

    # Removing NaN values
    model_df = model_df.replace(np.nan, '', regex=True)

    return model_df
