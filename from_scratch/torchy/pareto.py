import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def is_pareto(costs):
    # Boolean array: 1 is Pareto point, 0 otherwise
    is_efficient = np.ones(costs.shape[0], dtype=bool)  # Initially, all points are Pareto
    n_points = costs.shape[0]
    # For each data point
    for i in range(n_points):
        if is_efficient[i]:
            others = np.delete(costs, i, axis=0)
            others_efficient = np.delete(is_efficient, i, axis=0)

            dominated = np.any(np.all(others[others_efficient] <= costs[i], axis=1))

            is_efficient[i] = not dominated
    return is_efficient


plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})

def linear_ode_pareto():
    # Sigmoid dataframe
    d_sigmoid = {
        'Depth': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        'Neurons per layer': [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40],
        'Training loss': [0.67, 0.66, 0.576, 0.508, 2.55e-4, 9.71e-5, 1.18e-4, 2.2e-4, 5.14e-5, 7.61e-5, 6.77e-5, 5.34e-4, 2.17e-4, 2.34e-4, 5.76e-5, 1.22e-4, 1.27e-4, 6.54e-4, 1.2e-4, 3.69e-4],
        'Validation loss': [0.832, 0.824, 0.733, 0.657, 9.54e-4, 5.31e-4, 4.97e-4, 4.34e-4, 2.54e-4, 1.03e-3, 8.8e-4, 5.09e-4, 9.06e-4, 9.05e-4, 4.4e-4, 3.31e-4, 6.34e-4, 1.13e-3, 4.64e-4, 4.26e-4],
        'Training time': [14.23, 14.75, 14.81, 14.80, 19.27, 19.70, 19.67, 20.08, 23.95, 24.48, 24.65, 25.19, 28.56, 28.80, 28.75, 29.23, 33.07, 33.57, 35.45, 35.49]
    }

    df_sigmoid = pd.DataFrame(data=d_sigmoid)

    pareto_points_sigmoid = is_pareto(df_sigmoid[['Training time', 'Validation loss']].values)
    sigmoid_pareto = df_sigmoid[pareto_points_sigmoid]

    mask_sigmoid = np.ones_like(df_sigmoid, dtype=bool)
    mask_sigmoid[pareto_points_sigmoid] = False
    plot_sigmoid = df_sigmoid[mask_sigmoid]

    # Tanh
    d_tanh = {
        'Depth': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        'Neurons per layer': [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40],
        'Training loss': [0.145, 1.5e-2, 3.57e-2, 7.75e-3, 1.40e-4, 2.04e-4, 7.75e-5, 1.34e-4, 7.96e-5, 1.47e-5, 1.02e-4, 2.60e-4, 2.42e-4, 3.18e-4, 5.34e-5, 4.06e-3, 5.05e-4, 3.53e-5, 2.63e-4, 4.45e-4],
        'Validation loss': [0.208, 2.76e-2, 5.92e-2, 1.61e-2, 6.87e-4, 6.71e-4, 2.02e-3, 8.26e-4, 6.41e-4, 3.82e-4, 4.24e-4, 3.02e-4, 5.57e-4, 4.39e-4, 5.76e-4, 8.81e-3, 1.88e-3, 6.32e-4, 1.03e-3, 2.07e-3],
        'Training time': [15.43, 14.04, 14.13, 17.89, 26.24, 26.10, 26.16, 23.64, 27.43, 26.55, 26.95, 27.74, 33.61, 38.02, 36.47, 27.60, 35.23, 34.90, 31.83, 31.91]
    }

    df_sigmoid = pd.DataFrame(data=d_sigmoid)
    df_tanh = pd.DataFrame(data=d_tanh)

    df_stacked = pd.concat([df_sigmoid, df_tanh], axis=0)

    df_stacked = df_stacked.reset_index(drop=True)

    pareto_points = is_pareto(df_stacked[['Training time', 'Validation loss']].values)
    pareto = df_stacked[pareto_points]
    pareto_indices = np.array(pareto.index.tolist())

    mask_sigmoid = np.ones_like(df_sigmoid, dtype=bool)
    mask_sigmoid[pareto_points[:20]] = False
    plot_sigmoid = df_sigmoid[mask_sigmoid]

    mask_tanh = np.ones_like(df_tanh, dtype=bool)
    mask_tanh[pareto_points[20:]] = False
    plot_tanh = df_tanh[mask_tanh]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.scatter(plot_sigmoid['Training time'], plot_sigmoid['Validation loss'], s=(plot_sigmoid['Depth'] * 15),
               color='blue', label='Sigmoid non-Pareto points')
    ax.scatter(plot_tanh['Training time'], plot_tanh['Validation loss'], s=(plot_tanh['Depth'] * 15), marker='^',
               color='blue', label='Tanh non-Pareto points')
    ax.scatter(df_sigmoid['Training time'][pareto_points[:20]], df_sigmoid['Validation loss'][pareto_points[:20]],
               color='red', s=(df_sigmoid['Depth'].values[pareto_points[:20]] * 15), label='Sigmoid Pareto points')
    ax.scatter(df_tanh['Training time'][pareto_points[20:]], df_tanh['Validation loss'][pareto_points[20:]],
               color='red', s=(df_tanh['Depth'].values[pareto_points[20:]] * 15), marker='^',
               label='Tanh Pareto points')
    ax.set_xlabel('Training time (s)')
    ax.set_ylabel('Validation loss')
    ax.set_yscale('log')
    ax.set_title('Performance of network architectures for different activations')

    plt.legend(fontsize=14, loc='upper right')
    plt.show()


def nonlinear_ode_pareto():
    # Sigmoid dataframe
    d_sigmoid = {
        'Depth': [1, 1, 1, 1,
                  2, 2, 2, 2,
                  3, 3, 3, 3,
                  4, 4, 4, 4,
                  5, 5, 5, 5,
                  6, 6, 6, 6,
                  7, 7, 7, 7,
                  8, 8, 8, 8,
                  9, 9, 9, 9,
                  10, 10, 10, 10],
        'Neurons per layer': [10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40],
        'Training loss': [346.119, 141.127, 31.56, 51.65,
                          56.614, 0.1298, 8.14e-3, 3.25e-3,
                          281.6, 4.08e-3, 8.19e-4, 7.63e-4,
                          271.46, 2.572, 3.24e-3, 1.91e-3,
                          445.976, 6.499, 7.6e-3, 5.57e-3,
                          47.891, 5.818, 6.09e-2, 1.96e-2,
                          118.385, 68.271, 6.415, 7.62e-2,
                          102.137, 3.786, 38.646, 3.313,
                          716.180, 124.506, 7.715, 2.44e-2,
                          833.068, 103.67, 39.202, 8.735],
        'Validation loss': [828.345, 294.524, 37.12, 59.36,
                            134.200, 1.890, 16.204, 0.450,
                            1029.811, 25.974, 4.866, 1.751,
                            7225.26, 4880.43, 54.767, 8.339,
                            26901.72, 2208.715, 301.092, 2.470,
                            730.311, 3795.15, 1600.65, 99.174,
                            3716.27, 2273.974, 1275.360, 14.450,
                            3398.586, 3872.677, 470.335, 16.180,
                            3635.27, 3473.79, 3414.553, 6.217,
                            20202.64, 2383.84, 3523.38, 31.905],
        'Training time': [19.70, 20, 19.63, 19.51,
                          25.35, 27.59, 28.03, 33.50,
                          34.39, 35.09, 41.72, 38.79,
                          37.73, 38.05, 41.83, 42.25,
                          45.12, 46.38, 45.23, 45.60,
                          46.24, 52.07, 50.21, 79.63,
                          51.71, 58.57, 58.26, 62.60,
                          61.93, 62.85, 66.99, 67.90,
                          68.46, 72.06, 75.10, 75.44,
                          75.33, 74.34, 80.40, 109.28]
    }

    # Tanh
    d_tanh = {
        'Depth': [1, 1, 1, 1,
                  2, 2, 2, 2,
                  3, 3, 3, 3,
                  4, 4, 4, 4,
                  5, 5, 5, 5,
                  6, 6, 6, 6,
                  7, 7, 7, 7,
                  8, 8, 8, 8,
                  9, 9, 9, 9,
                  10, 10, 10, 10],
        'Neurons per layer': [10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40,
                              10, 20, 30, 40],
        'Training loss': [50.26, 10.204, 0.912, 9.774,
                          0.2507, 7.19e-3, 1.31e-2, 3.75e-2,
                          1.79e-2, 1.58e-2, 3.03e-3, 1.20e-2,
                          1.24e-2, 3.66e-2, 1.56e-4, 0.166,
                          5.15e-3, 0.11, 7.13e-2, 0.137,
                          0.1143, 1.38e-2, 0.679, 6.48e-3,
                          34.573, 2.28e-2, 4.39e-3, 2.92e-3,
                          129.476, 1.73e-3, 0.2113, 1.71e-2,
                          155.646, 0.3429, 1.36e-2, 1.1,
                          90.817, 4.49e-2, 0.696, 6.17e-3],
        'Validation loss': [58.58, 12.596, 2.069, 11.838,
                            1.004, 0.3731, 0.275, 0.719,
                            2.834, 3.14, 3.025, 1.053,
                            12.09, 2.285, 1.117, 1.262,
                            3.395, 1.172, 0.504, 0.296,
                            1362.28, 3.721, 1.3, 3.924,
                            90.6, 562.95, 7.981, 0.2734,
                            4445.346, 0.6278, 1.275, 16.634,
                            883.667, 0.9018, 2.671, 0.297,
                            226.5, 1.499, 0.4448, 6.11],
        'Training time': [17.1, 15.98, 16.22, 22.71,
                          26.41, 26.55, 29.09, 28.16,
                          32.05, 32.13, 36.89, 37.71,
                          29.91, 30.18, 31.51, 33.43,
                          37.09, 38.43, 39.26, 38.14,
                          41.09, 40.88, 43.93, 44.68,
                          56.66, 42.75, 60.74, 61.36,
                          63.19, 63.04, 63.74, 65.8,
                          80.25, 81.86, 66.64, 73.76,
                          71.8, 72.98, 99.25, 75.54]
    }

    df_sigmoid = pd.DataFrame(data=d_sigmoid)
    df_tanh = pd.DataFrame(data=d_tanh)

    df_stacked = pd.concat([df_sigmoid, df_tanh], axis=0)

    df_stacked = df_stacked.reset_index(drop=True)

    pareto_points = is_pareto(df_stacked[['Training time', 'Validation loss']].values)
    pareto = df_stacked[pareto_points]
    pareto_indices = np.array(pareto.index.tolist())

    mask_sigmoid = np.ones_like(df_sigmoid, dtype=bool)
    mask_sigmoid[pareto_points[:40]] = False
    plot_sigmoid = df_sigmoid[mask_sigmoid]

    mask_tanh = np.ones_like(df_tanh, dtype=bool)
    mask_tanh[pareto_points[40:]] = False
    plot_tanh = df_tanh[mask_tanh]

    fig, ax = plt.subplots(1, 1, figsize=(10,7))

    ax.scatter(plot_sigmoid['Training time'], plot_sigmoid['Validation loss'], s=(plot_sigmoid['Depth']*10), color='blue', label='Sigmoid non-Pareto points')
    ax.scatter(plot_tanh['Training time'], plot_tanh['Validation loss'], s=(plot_tanh['Depth']*10), marker='^', color='blue', label='Tanh non-Pareto points')
    ax.scatter(df_sigmoid['Training time'][pareto_points[:40]], df_sigmoid['Validation loss'][pareto_points[:40]], color='red', s=(df_sigmoid['Depth'].values[pareto_points[:40]]*10), label='Sigmoid Pareto points')
    ax.scatter(df_tanh['Training time'][pareto_points[40:]], df_tanh['Validation loss'][pareto_points[40:]], color='red', s=(df_tanh['Depth'].values[pareto_points[40:]]*10), marker='^', label='Tanh Pareto points')
    ax.set_xlabel('Training time (s)')
    ax.set_ylabel('Validation loss')
    ax.set_yscale('log')
    ax.set_title('Performance of network architectures for different activations')

    plt.legend(fontsize=14, loc='upper right')
    plt.show()

def pde_pareto():
    # Only tanh is used
    d = {
        'Depth': [5, 5, 5,
                  6, 6, 6,
                  7, 7, 7,
                  8, 8, 8,
                  9, 9, 9,
                  10, 10, 10],
        'Neurons per layer': [20, 30, 40,
                              20, 30, 40,
                              20, 30, 40,
                              20, 30, 40,
                              20, 30, 40,
                              20, 30, 40],
        'Training loss': [0.275, 0.278, 0.449,
                          0.728, 0.577, 0.398,
                          1.048, 0.163, 0.787,
                          0.1213, 0.177, 0.402,
                          0.803, 0.201, 8.056,
                          1.3415, 0.3796, 0.4123],
        'Validation loss': [0.436, 0.412, 0.701,
                            0.132, 0.890, 0.595,
                            1.00, 0.276, 3.566,
                            0.843, 0.286, 7.232,
                            0.429, 0.280, 4.598,
                            1.984, 1.574, 0.370],
        'Training time': [340.03, 255.44, 164.88,
                          488.17, 312.00, 199.24,
                          706.61, 507.83, 321.01,
                          563.25, 420.38, 256.62,
                          760.70, 532.12, 434.23,
                          785.10, 465.72, 369.39]
    }

    df = pd.DataFrame(data=d)

    pareto_points = is_pareto(df[['Training time', 'Validation loss']].values)
    pareto = df[pareto_points]
    pareto_indices = np.array(pareto.index.tolist())

    mask = np.ones_like(df, dtype=bool)
    mask[pareto_points] = False
    df_nonpareto = df[mask]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.scatter(df_nonpareto['Training time'], df_nonpareto['Validation loss'], s=(df_nonpareto['Depth'] * 8),
               color='blue', label='Non-Pareto points')
    ax.scatter(df['Training time'][pareto_points], df['Validation loss'][pareto_points],
               color='red', s=(df['Depth'].values[pareto_points] *8), label='Pareto points')

    ax.set_xlabel('Training time (s)')
    ax.set_ylabel('Validation loss')
    ax.set_yscale('log')
    ax.set_title('Performance of network architectures for tanh activation')

    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.show()

pde_pareto()