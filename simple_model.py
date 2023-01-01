# simple model with drift

# libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# r_0: initial interest rate
# mu: drift term (if =0, simple model itself!)
# sig: sigma term
# t: time
# dt: time length
# rnp: risk neutral probability


def sm(r_0, mu, sig, t, dt, rnp):

    # Organizing simple model

    # DataFrame to hold interest rate tree
    model_df = pd.DataFrame(index=range(0, t), columns=range(0, t))

    # Organizing trees
    for i in range(t):
        model_df.loc[0, i] = round(r_0 + i * mu * dt + i * sig * np.sqrt(dt),
                                   4)

    for k in range(1, t):
        for j in range(1, t):
            if model_df.columns[k] < model_df.index[j]:
                pass
            else:
                model_df[k][j] = round(
                    model_df[k][0] - 2 * j * sig * np.sqrt(dt), 4)

    # State price (Forward recursion)

    state_df = model_df.copy().replace(np.nan, 0, regex=True)

    refer_df = state_df.copy()

    state_df.iloc[0, 0] = 1

    for i in range(1, t):
        state_df.loc[0, i] = rnp * state_df.loc[0, i - 1] / (
            1 + dt * model_df.loc[0, i - 1])

    for j in range(1, t):
        for k in range(1, t):
            state_df.loc[j, k] = rnp * state_df.loc[j - 1, k - 1] / (
                1 + dt * refer_df.loc[j - 1, k - 1]
            ) + (1 - rnp) * state_df.loc[j, k -
                                         1] / (1 + dt * refer_df.loc[j, k - 1])

    # Discount factor / Spot rate / Forward rate / Par rate

    d_t = [round(state_df.loc[:, i].sum(), 4) for i in range(t)]

    r_t = [
        round((1 / dt) * ((1 / (d_t[i])**(1 / i)) - 1), 4)
        for i in range(1, t)
    ]

    fwd_t = [
        round((1 / dt) * (d_t[i - 1] / d_t[i] - 1), 4) for i in range(1, t)
    ]

    par_t = [
        round((1 / dt) * (1 - d_t[i]) / np.sum(d_t[1:i + 1]), 4)
        for i in range(1, t)
    ]

    fwd_t[0] = r_t[0]
    r_t.insert(0, np.nan)
    fwd_t.insert(0, np.nan)
    par_t.insert(0, np.nan)

    rate_df = pd.DataFrame({
        'discount factor': d_t,
        'spot rate': r_t,
        'fwd rate': fwd_t,
        'par rate': par_t
    }).T

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
    plt.title('Simple model with drift')
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
            if type(k) == float:
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

    return model_df, state_df, rate_df
