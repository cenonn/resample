import numpy as np
import matplotlib.pyplot as plt


class UnivariateResult:
    """Defines an object for univariate bootstraps

            :param results: array that contains the estimate \
                for every bootstrap replication
            :param statistic: statistic to be calculated
            :param observed: the sample statistic from the original data
            :type results: np.array
            :type statistic: function
            :type observed: int, float
    """
    def __init__(self, results, statistic, observed):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed

    def se(self):
        """Bootstrap estimate of standard error

            :return: standard error
            :rtype: float
        """
        return self.results.std(ddof=1)

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
            return center(self.results)
        else:
            return center(self.results) - self.bias()

    def plot(self, bins=30):
        """Create a histogram of the bootstrap distribution

            :param bins: number of bins for the histogram
            :type bins: int
        """
        bins = int(bins)
        plt.hist(self.results, bins=bins)

    def ci(self, confidence=0.95, kind='efron'):
        """Calculate an interval estimate of the statistic

            :param confidence: the confidence level of the estimate, \
                between 0.0 and 1.0e con
            :param kind: type of interval to calculate, either efron or \
                percentile intervals
            :type confidence: float
            :type kind: string
            :return: the confidence interval
            :rtype: tuple

        """
        quantile = 100*(1 - confidence)/2
        res = self.results
        L, U = np.percentile(res, quantile), np.percentile(res, 100 - quantile)
        if kind == 'efron':
            return L, U
        elif kind == 'loc_percentile':
            return 2*self.observed - U, 2*self.observed - L
        elif kind == 'scale_percentile':
            return self.observed ** 2 / U, self.observed ** 2 / L
        else:
            raise Exception("unsupported ci type")


class MultivariateResult:
    """Defines an object for multivariate bootstraps

            :param results: array that contains the estimate \
                for every bootstrap replication
            :param statistic: statistic to be calculated
            :param observed: the sample statistics from the original data
            :type results: np.array
            :type statistic: function
            :type observed: np.array
    """
    def __init__(self, results, statistic, observed):
        self.results = np.array(results)
        self.statistic = statistic
        self.observed = observed

    def se(self):
        """Bootstrap estimate of standard error for all variables

            :return: standard error
            :rtype: float
        """
        return np.apply_along_axis(np.std, 0, self.results, ddof=0)

    def bias(self):
        """Bootstrap estimate of bias for all variables

            :return: bias
            :rtype: float
        """
        """A function that takes no arguments and returns an array
            that contains the bias of the statistic for all variables.
        """
        return np.mean(self.results) - self.observed

    def point_estimate(self, center, correction=False):
        """Point estimate from the bootstrap distribution

            :param center: measure of center to use
            :param correction: if True, apply bias correction
            :type center: function
            :type correction: boolean
            :return: numerical estimate of center
            :rtype: np.array

        """
        if not correction:
            return np.apply_along_axis(center, 0, self.results)
        else:
            return np.apply_along_axis(center, 0, self.results) - self.bias()

    def plot(self, col, bins=30):
        """Create a histogram of the bootstrap distribution

            :param bins: number of bins for the histogram
            :param col: index of the variable to plot
            :type bins: int
            :type col: int
        """
        col = int(col)
        bins = int(bins)
        plt.hist(self.results[:,col], bins=bins)

    def ci(self, col, confidence=0.95, kind='efron'):
        """Calculate an interval estimate of the statistic

            :param col: index of the variable
            :param confidence: the confidence level of the estimate, \
                between 0.0 and 1.0e con
            :param kind: type of interval to calculate, either efron or \
                percentile intervals
            :type col: int
            :type confidence: float
            :type kind: string
            :return: the confidence interval
            :rtype: tuple

        """
        col = int(col)
        quantile = 100*(1 - confidence)/2
        res = self.results[:,col]
        L, U = np.percentile(res, quantile), np.percentile(res, 100 - quantile)
        if kind == 'efron':
            return L, U
        elif kind == 'loc_percentile':
            return 2*self.observed - U, 2*self.observed - L
        elif kind == 'scale_percentile':
            return self.observed ** 2 / U, self.observed ** 2 / L
        else:
            raise Exception("unsupported ci type")
