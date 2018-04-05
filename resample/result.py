import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

# convert to just one overarching Result object
# have the functions below just return what ever is outputted by the estimator

class Results:
    """Defines an object for bootstrap results

            :param results: array that contains the estimate \
                for every bootstrap replication
            :param statistic: statistic to be calculated
            :param observed: the sample statistic from the original data
            :param data: the original data
            :type results: np.array
            :type statistic: function
            :type observed: int, float
            :type data: np.array, pd.Series, pd.DataFrame
    """
    def __init__(self, results, statistic, observed, data):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed
        self.data = np.array(data)

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

    def point_estimate(self, center, correction=False):
        """Point estimate from the bootstrap distribution

            :param center: measure of center to use
            :param correction: if True, apply bias correction
            :type center: function
            :type correction: boolean
            :return: numberical estimate of center
            :rtype: float, int

        """
        if not correction:
            return np.apply_along_axis(center, 0, self.results)
        else:
            return np.apply_along_axis(center, 0, self.results) - self.bias()

    def plot(self, col=None, row=None, bins=30):
        """Create a histogram of the bootstrap distribution

            :param bins: number of bins for the histogram
            :param col: index of the variable to plot
            :type bins: int
            :type col: int
        """
        if col is None and row is None:
            plt.hist(self.results, bins=bins)
        elif col is not None and row is None:
            plt.hist(self.results[col], bins=bins)
        elif col is not None and row is not None:
            plt.hist(self.results[:, row, col], bins=bins)
        else:
            raise Exception("please provide at least column to plot")

    def ci(self, col=None, row=None, confidence=0.95, kind='efron'):
        """Calculate an interval estimate of the statistic

            :param confidence: the confidence level of the estimate, \
                between 0.0 and 1.0
            :param kind: type of interval to calculate, either efron or \
                percentile intervals
            :type confidence: float
            :type kind: string
            :return: the confidence interval
            :rtype: tuple
        """
        if col is None and row is None:
            res = self.results
            obs = self.observed
        elif col is not None and row is None:
            res = self.results[:, col]
            obs = self.observed[col]
        else:
            res = self.results[:, row, col]
            obs = self.observed[row, col]

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
            z_hat_nought = norm.ppf(sum(res < obs) / len(res))

            # calculate acceleration
            index_range = np.arange(0, len(self.data))
            jack_index = (np.delete(index_range, i) for i in index_range)
            theta_i = [np.mean(self.data[i]) for i in jack_index]
            theta_dot = np.mean(theta_i)
            a_hat_num = np.sum((theta_dot - theta_i) ** 3)
            a_hat_den = 6.0 * np.sum((theta_dot - theta_i) ** 2) ** 1.5
            a_hat = a_hat_num / a_hat_den

            # calculate the endpoints
            a1_num = z_hat_nought + norm.ppf(1 - confidence)
            a1_den = 1 - a_hat * (z_hat_nought + norm.ppf(1 - confidence))
            a2_num = z_hat_nought + norm.ppf(confidence)
            a2_den = 1 - a_hat * (z_hat_nought + norm.ppf(confidence))
            a1 = 100 * norm.cdf(z_hat_nought + (a1_num / a1_den))
            a2 = 100 * norm.cdf(z_hat_nought + (a2_num / a2_den))
            return np.percentile(res, a1), np.percentile(res, a2)
        else:
            raise Exception("unsupported ci type")

