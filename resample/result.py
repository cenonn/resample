import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from .utility import group_res, output_res, bca_endpoints, \
                        plot_single


class Results:
    """Defines an object for bootstrap results

        :param results: array that contains the estimate \
            for every bootstrap replication
        :param statistic: statistic that was calculated
        :param observed: the sample statistic from the original data
        :param data: the original data
        :param group_cols: group columns for data, default is none
        :param output_cols: output columns for data, default is none
        :type results: np.array
        :type statistic: function
        :type observed: int, float, np.array
        :type data: np.array, pd.Series, pd.DataFrame
        :type group_cols: list
        :type output_cols: list
    """
    def __init__(self, results, statistic, observed, data, group_cols=None, \
                                output_cols=None):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed
        self.data = data
        self.group_cols = group_cols
        self.output_cols = output_cols
        self.ndim = self.results.ndim
        self.shape = self.results.shape

    def se(self):
        """Bootstrap estimate of standard error

            :return: standard error
            :rtype: float
        """
        return np.apply_along_axis(np.std, 0, self.results)

    def bias(self):
        """Bootstrap estimate of bias

            :return: bias
            :rtype: float
        """
        return np.mean(self.results) - self.observed

    def point_estimate(self, center, correction=False, **kwargs):
        """Point estimate from the bootstrap distribution

            :param center: measure of center to use
            :param correction: if True, apply bias correction
            :type center: function
            :type correction: boolean
            :return: numberical estimate of center
            :rtype: np.array

        """
        center_func = lambda x: center(x, **kwargs)
        if not correction:
            return np.apply_along_axis(center_func, 0, self.results)
        else:
            return np.apply_along_axis(center_func, 0, self.results) - self.bias()

    def plot(self, col=None, row=None, bins=30, figsize=(6, 4), **kwargs):
        """Create histograms of the bootstrap distribution

            :param col: y index of the variable to plot
            :param row: x index of the variable to plot (requires col)
            :param bins: number of bins for the histogram
            :param figsize: size of figure in inches
            :type col: int
            :type row: int
            :type bins: int
            :type figsize: tuple
        """
        res_shape = len(self.shape) #determines how many histrograms to make
        if col is None and row is None: #make all possible histograms
            if res_shape == 1:
                plt.figure(figsize=figsize)
                plt.hist(self.results, bins=bins, **kwargs)
            elif res_shape == 2:
                num_plots = self.shape[1]
                plot_single(self.results, num_plots, bins, figsize, **kwargs)

            elif res_shape == 3:
                x_plots = self.shape[1]
                y_plots = self.shape[2]
                fig, axes = plt.subplots(x_plots, y_plots,
                                                figsize=figsize, sharey=True)
                axes_iter = np.nditer(axes, flags=["refs_ok"])

                for x in range(x_plots):
                    for y in range(y_plots):
                        current_var = self.results[:, x, y]
                        axes_iter[0].item(0).hist(current_var, \
                                                    bins=bins, **kwargs)
                        axes_iter.iternext()

        elif col is not None and row is None:
            if res_shape == 2:
                plt.hist(self.results[:, col], figsize=figsize,
                                                    bins=bins, **kwargs)
            elif res_shape == 3:
                col_vals = self.results[:, :, col]
                num_plots = col_vals.shape[1]
                plot_single(col_vals, num_plots, bins, figsize, **kwargs)

        elif col is None and row is not None:
            raise Exception("provide column to plot")
        elif col is not None and row is not None:
            plt.hist(self.results[:, row, col], bins=bins, **kwargs)

    def ci(self, col=None, row=None, confidence=0.95, kind='efron'):
        """Calculate an interval estimate of the statistic

            Efron 'efron':
            :math:`(\\hat{\\theta}^{*(\\alpha)},\\hat{\\theta}^{*(1-\\alpha)})`

            Location percentile 'loc_percentile':
            :math:`(2\\theta-U, 2\\theta-L)`

            Scale percentile 'scale_percentile':
            :math:`(\\frac{\\theta^2}{U}, \\frac{\\theta^2}{L})`

            Bias corrected and accelerated 'BCa':
            :math:`(\\hat{\\theta}^{*(\\alpha_1)},\\hat{\\theta}^{*(\\alpha_2)})`

            where

            :math:`\\alpha_1=\\Phi(\\hat{z_0}+\\frac{\\hat{z_0}+z^{(\\alpha)}}{1-\\hat{\\alpha}(\\hat{z_0}+z^{(\\alpha)})})`

            :math:`\\alpha_2=\\Phi(\\hat{z_0}+\\frac{\\hat{z_0}+z^{(1-\\alpha)}}{1-\\hat{\\alpha}(\\hat{z_0}+z^{(1-\\alpha)})})`

            and

            :math:`\\hat{z}_0=\\Phi^{-1}(\\frac{\#\{\\hat{\\theta}^*(b)<\\hat{\\theta}\}}{B})`

            :math:`\\hat{\\alpha}=\\frac{\\sum_{i=1}^n(\\hat{\\theta}_{(.)}-\\hat{\\theta}_{(i)})^3}{6\{\\sum_{i=1}^n(\\hat{\\theta}_{(.)}-\\hat{\\theta}_{(i)})^2\}^{3/2}}`

            :math:`.`

            :param col: column to plot
            :param row: row to plot
            :param confidence: the confidence level of the estimate, \
                between 0.0 and 1.0
            :param kind: type of interval to calculate, either efron, \
                BCa, location percentile, or scale percentile
            :type col: int
            :type row: int
            :type confidence: float
            :type kind: string
            :return: the confidence interval
            :rtype: tuple
        """
        data = self.data
        if col is None and row is None:
            res = self.results
            obs = self.observed
        elif col is not None and row is None:
            res = self.results[:, col]
            obs = self.observed[col]
        elif col is not None and row is not None:
            res = self.results[:, row, col]
            obs = self.observed[row, col]
        else:
            raise Exception("column argument needs to be specified to use row argument")

        quantile = 100*(1 - confidence)/2
        L, U = np.percentile(res, quantile), np.percentile(res, 100 - quantile)

        if kind == 'efron':
            return L, U
        elif kind == 'loc_percentile':
            return 2*obs - U, 2*obs - L
        elif kind == 'scale_percentile':
            return obs ** 2 / U, obs ** 2 / L
        elif kind == 'BCa':
            # calculate bias-correction
            z_hat_nought = norm.ppf((res < obs).mean())
            if np.isinf(z_hat_nought):
                raise Exception("bias-correction is inf. try raising value of r")

            # calculate acceleration
            if isinstance(data, pd.DataFrame):
                theta_i = []
                #perform jackknife
                for i, _ in data.iterrows():
                    current_iter = data.drop(i)
                    if self.group_cols is None:
                        if self.output_cols is None:
                            theta_i.append(self.statistic(current_iter))
                        else:
                            X, y = output_res(current_iter, self.output_cols)
                            if col is None and row is None:
                                current_res = self.statistic(X, y)
                            elif col is not None and row is None:
                                current_res = self.statistic(X, y)[col]
                            else:
                                current_res = self.statistic(X, y)[row, col]
                            theta_i.append(current_res)
                    else:
                        current_res, _ = group_res(current_iter,
                                            self.group_cols, self.statistic)
                        theta_i.append(current_res)
            else:
                index_range = np.arange(0, len(data))
                jack_index = (np.delete(index_range, i) for i in index_range)
                theta_i = [self.statistic(data[i]) for i in jack_index]

            theta_dot = np.mean(theta_i)
            a_hat_num = np.sum((theta_dot - theta_i) ** 3)
            a_hat_den = 6.0 * np.sum((theta_dot - theta_i) ** 2) ** 1.5
            a_hat = a_hat_num / a_hat_den

            # calculate the endpoints
            a1 = bca_endpoints(z_hat_nought, a_hat, 1 - confidence)
            a2 = bca_endpoints(z_hat_nought, a_hat, confidence)
            return np.percentile(res, a1), np.percentile(res, a2)
        else:
            raise Exception("unsupported ci type")

