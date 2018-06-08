import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


def group_res(data, group_cols, statistic):
    """Splits dataframe into dictionary based on grouping

        :param data: input data to be split
        :param group_cols: group columns for data
        :param statistic: statistic to be calculated
        :type data: pd.DataFrame
        :type group_cols: list
        :type statistic: function
        :return: results from the grouping and the grouped data
        :rtype: tuple

    """
    indices = (data.reset_index()
                .groupby(group_cols)["index"]
                .apply(list).to_dict())
    grouped_data = {}
    for key, val in indices.items():
        grouped_data[key] = (data
            .loc[val][data.loc[val]
            .columns.difference(group_cols)])
    current_res = statistic(*list(grouped_data.values()))
    return current_res, grouped_data

def output_res(data, output_cols):
    """Splits dataframe into X and y inputs

        :param data: input data to be split
        :param output_cols: output columns for data, default is none
        :type data: pd.DataFrame
        :type output_cols: list
        :return: the dataframe split into X and y
        :rtype: tuple

    """
    X = data[list(data.columns.difference(output_cols))]
    if len(output_cols) == 1:
        y = data[output_cols[0]]
    else:
        y = data[output_cols]
    return X, y

def bca_endpoints(z_hat_nought, a_hat, percentile):
    """Calculate an endpoint for BCa

        :param z_hat_nought: bias correction
        :param a_hat: acceleration component
        :param percentile: percentile for the endpoint
        :type z_hat_nought: float
        :type a_hat: float
        :type percentile: float
        :return: the percentile value
        :rtype: float

    """
    num = z_hat_nought + norm.ppf(percentile)
    den = 1 - a_hat * (z_hat_nought + norm.ppf(percentile))
    a = 100 * norm.cdf(z_hat_nought + (num / den))
    return a

def plot_single(data, num_plots, bins, figsize, **kwargs):
    """Create set of plots

        :param data: values to plot
        :param num_plots: number of plots
        :param bins: number of bins for the histogram
        :type data: np.array
        :type num_plots: int
        :type bins: int

    """
    fig, axes = plt.subplots(num_plots, figsize=figsize, sharey=True)

    for ax, i in zip(axes, range(0, num_plots)):
        current_var = data[:,i]
        ax.hist(current_var, bins=bins, **kwargs)

