import numpy as np
import pandas as pd

from .metric import Metric


def boot(data, statistic, r=10000):
    results = []
    sample_size = len(data)
    if isinstance(data, np.ndarray):
        for _ in range(r):
            boot_sample = [data[i] for i in np.random.randint(sample_size, size=sample_size)]
            results.append(statistic(boot_sample))
        return Metric(results, statistic)
    elif isinstance(data, pd.DataFrame): 
        for _ in range(r):
            boot_sample = [data[0][i] for i in np.random.randint(sample_size, size=sample_size)]
            results.append(statistic(boot_sample))
        return Metric(results, statistic)
    elif isinstance(data, pd.Series): 
        for _ in range(r):
            boot_sample = [data[i] for i in np.random.randint(sample_size, size=sample_size)]
            results.append(statistic(boot_sample))
        return Metric(results, statistic)

