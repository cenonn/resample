import numpy as np
import numbers


class Statistic:
    """Defines a Statistic object with user defined function.

            :param func: statistic to be calculated
            :param is_valid: whether the function is valid to bootstrap
            :type func: function
            :type is_valid: boolean
    """

    def __init__(self, func, is_valid=False):
        self.func = func
        self.is_valid = is_valid

    def __add__(self, other):
        if isinstance(other, Statistic):
            valid = self.is_valid and other.is_valid
            return Statistic(lambda x: self.func(x) + other.func(x), valid)
        elif isinstance(other, numbers.Numbers):
            return Statistic(lambda x: self.func(x) + other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Statistic):
            valid = self.is_valid and other.is_valid
            return Statistic(lambda x: self.func(x) - other.func(x), valid)
        elif isinstance(other, numbers.Numbers):
            return Statistic(lambda x: self.func(x) - other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Statistic):
            valid = self.is_valid and other.is_valid
            return Statistic(lambda x: self.func(x) * other.func(x), valid)
        elif isinstance(other, numbers.Numbers):
            return Statistic(lambda x: self.func(x) * other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Statistic):
            valid = self.is_valid and other.is_valid
            return Statistic(lambda x: self.func(x) / other.func(x), valid)
        elif isinstance(other, numbers.Numbers):
            return Statistic(lambda x: self.func(x) / other, self.is_valid)
        else:
            raise Exception("other is of type " + string(type(other)))

    def __rtruediv__(self, other):
        return self.__truediv__(other)


class Mean(Statistic):
    """Defines valid Statistic object with a mean function.

        :param none:

    """
    def __init__(self):
        super().__init__(np.mean, True)


class Quantile(Statistic):
    """Defines Statistic object either the median or quantile.

        :param q: quantile to be calculated
        :type q: float
    """
    def __init__(self, q):
        if (q > 1.0 or q < 0.0):
            raise Exception("q must be between 0.0 and 1.0")
        if (q == 0.5):
            super().__init__(np.median, True)
        else:
            super().__init__(lambda x: np.percentile(x, q * 100), False)


class Median(Quantile):
    def __init__(self):
        super().__init__(0.5)


class SD(Statistic):
    """Defines valid Statistic object with a sd function.

        :param none:

    """
    def __init__(self):
        super().__init__(np.std, True)


class Var(Statistic):
    """Defines valid Statistic object with a variance function.

        :param none:

    """
    def __init__(self):
        super().__init__(np.var, True)


class Max(Statistic):
    """Defines valid Statistic object with a max function.

        :param none:

    """
    def __init__(self):
        super().__init__(np.amax, False)


class Min(Statistic):
    """Defines valid Statistic object with a min function.

        :param none:

    """
    def __init__(self):
        super().__init__(np.amin, False)
