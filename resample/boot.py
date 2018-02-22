import numpy as np
import pandas as pd

from .result import UnivariateResult
from .result import MultivariateResult
from .statistic import Statistic


def boot(data, statistic, r=10000):
    """Creates UnivariateResult or MultivariateResult objects.

        :param data: input data to be bootstrapped
        :param statistic: statistic to be calculated
        :param r: number of bootstrap replications
        :type data: np.array, pd.Series, np.DataFrame
        :type statistic: Statistic, function
        :type t: int
        :return: results from the bootstrap simulation
        :rtype: UnivariateResult, MultivariateResult

    """
    if isinstance(statistic, Statistic):
        func = statistic.func
        if not statistic.is_valid:
            raise Warning("results from bootstrap may not be valid")
    else:
        func = statistic
    r = int(r)  # make sure r is an int
    results = []
    sample_size = len(data)
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            for _ in range(r):
                boot_sample = np.random.choice(data, sample_size, replace=True)
                results.append(func(boot_sample))
            return UnivariateResult(results, func, func(data))
        else:
            raise Exception("numpy array must contain only 1 variable")
    elif isinstance(data, pd.Series):
        for _ in range(r):
            boot_sample = data.sample(n=sample_size, replace=True)
            results.append(func(boot_sample))
        return UnivariateResult(results, func, func(data))
    elif isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            for _ in range(r):
                boot_sample = data.sample(n=sample_size, replace=True).as_matrix()
                results.append(func(boot_sample))
            return UnivariateResult(results, func, func(data))
        else:
            for _ in range(r):
                boot_sample = data.sample(n=sample_size, replace=True)
                results.append(boot_sample.apply(func))
            return MultivariateResult(results, func, np.apply_along_axis(func, 0, data))
    else:
        raise Exception("data type not supported")


