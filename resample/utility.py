import pandas as pd
import numpy as np

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

def bca_endpoints(z_hat_nought, a_hat, conf):
    """Calculate an endpoint for BCa

        :param z_hat_nought: bias correction
        :param a_hat: acceleration component
        :param conf: confidence value for the endpoint
        :type z_hat_nought: float
        :type a_hat: float
        :type conf: float
        :return: the percentile value
        :rtype: float

    """
    num = z_hat_nought + norm.ppf(conf)
    den = 1 - a_hat * (z_hat_nought + norm.ppf(conf))
    a = 100 * norm.cdf(z_hat_nought + (num / den))
    return a