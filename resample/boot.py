import numpy as np
import pandas as pd

from .result import Results
from .statistic import Statistic


def boot(data, statistic, group_cols=None, output_cols=None, r=10000, **kwargs):
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
                results.append(func(boot_sample, **kwargs))
            return Results(results, func, func(data, **kwargs), data)
        else:
            raise Exception("numpy array must contain only 1 variable")
    elif isinstance(data, pd.Series):
        for _ in range(r):
            boot_sample = data.sample(n=sample_size, replace=True)
            results.append(func(boot_sample, **kwargs))
        return Results(results, func, func(data, **kwargs), data)
    elif isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            for _ in range(r):
                boot_sample = data.sample(n=sample_size, replace=True).as_matrix()
                results.append(func(boot_sample, **kwargs))
            return Results(results, func, func(data, **kwargs), data)
        else:
            if group_cols is None:
                for _ in range(r):
                    # frac=1 to return sample of same size
                    boot_sample = data.sample(frac=1, replace=True)
                    if output_cols is None:
                        results.append(func(boot_sample, **kwargs))
                    else:
                        results.append(func(boot_sample[boot_sample.columns.difference(output_cols)], boot_sample[output_cols],**kwargs))
                if output_cols is None:
                    return Results(results, func, func(data, **kwargs), data)
                else:
                    return Results(results, func, func(boot_sample[boot_sample.columns.difference(output_cols)], boot_sample[output_cols],**kwargs), data)
            else:
                # TODO
                indices = data.reset_index().groupby(group_cols)["index"].apply(list).to_dict()
                grouped_data = {}
                for key, val in indices.items():
                    grouped_data[key] = data.iloc[val][data.iloc[val].columns.difference(group_cols)]

                for _ in range(r):
                    current_res = []
                    for key, val in grouped_data.items():
                        boot_sample = val.sample(frac=1, replace=True)
                        current_res.append(boot_sample)
                    results.append(func(*current_res, **kwargs).as_matrix())
                return Results(results, func, func(*list(grouped_data.values()), **kwargs).as_matrix(), grouped_data)
    else:
        raise Exception("data type not supported")


