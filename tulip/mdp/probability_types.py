# To hold the different types of probability for use in UncertainMDP's.
#
# Originally: James Bern 8/17/2013
# jbern@caltech.edu
#
# class float:
#    """
#    builtin float
#    """

class IntervalProbability:
    """
    A single interval of probability for use in an UncertainMDP.
    """
    def __init__(self, low, high):
        # Case 2: a finite set of ranges of probabilities
        assert type(low) is float
        assert type(high) is float
        assert low <= high, "not of form (this-float-less-than, this-float)"
        self.low = low
        self.high = high
        self.interval = (low, high)
        assert self.low <= 1.0 and self.low >= 0.0, str(self.low) + " is invalid low"
        assert self.high <= 1.0 and self.high >= 0.0, str(self.high) + " is invalid high"

    def __repr__(self):
        """
        For easier reading of output return a different format for trivial intervals.
        """
        if self.low != self.high:
            return "<%s, %s>" % (self.low, self.high)
        else:
            return "<%s>" % self.low

    def __eq__(self, other):
        return self.interval == other.interval

    def __ne__(self, other):
        return self.interval != other.interval

    def __add__(self, other):
        """
        Standard interval addition function with a ceiling at 1.0 for low and high.
        """
        result = (self.low + other.low, self.high + other.high)
        return IntervalProbability(min(1.0, result[0]), min(1.0, result[1]))

    @staticmethod
    def zero():
        """
        Generate a trivial 0.
        """
        return IntervalProbability(0.0, 0.0)

    @staticmethod
    def one():
        """
        Generate a trivial 1.
        """
        return IntervalProbability(1.0, 1.0)

    def contains(self, p):
        """
        Returns whether an input p is in the range [low, high].
        NOTE: have not used this function yet.
        """
        low, high = self.interval
        return p >= low and p <= high


