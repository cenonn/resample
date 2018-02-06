import numpy as np


class Statistic:
    """Defines a user defined Statistic object.

    Args:
        func (function): statistic to be calculated
        is_valid (boolean): whether the function is valid to bootstrap
    """
    def __init__(self, func, is_valid=False):
        self.func = func
        self.is_valid = is_valid

    def __add__(self, other):
        if isinstance(other, Statistic):
            valid = True if self.is_valid and other.is_valid else False
            return Statistic(lambda x: self.func(x) + other.func(x), valid)
        elif isinstance(other, (int, float)):
            return Statistic(lambda x: self.func(x) + other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Statistic):
            valid = True if self.is_valid and other.is_valid else False
            return Statistic(lambda x: self.func(x) - other.func(x), valid)
        elif isinstance(other, (int, float)):
            return Statistic(lambda x: self.func(x) - other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Statistic):
            valid = True if self.is_valid and other.is_valid else False
            return Statistic(lambda x: self.func(x) * other.func(x), valid)
        elif isinstance(other, (int, float)):
            return Statistic(lambda x: self.func(x) * other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Statistic):
            valid = True if self.is_valid and other.is_valid else False
            return Statistic(lambda x: self.func(x) / other.func(x), valid)
        elif isinstance(other, (int, float)):
            return Statistic(lambda x: self.func(x) / other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __rtruediv__(self, other):
        return self.__truediv__(other)

class MeanStatistic(Statistic):
    def __init__(self):
        super().__init__(np.mean, True)

class MedianStatistic(Statistic):
    def __init__(self):
        super().__init__(np.median, True)

class SDStatistic(Statistic):
    def __init__(self):
        super().__init__(np.std, True)

class VarStatistic(Statistic):
    def __init__(self):
        super().__init__(np.var, True)

class MaxStatistic(Statistic):
    def __init__(self):
        super().__init__(np.amax, False)

class MinStatistic(Statistic):
    def __init__(self):
        super().__init__(np.amin, False)
