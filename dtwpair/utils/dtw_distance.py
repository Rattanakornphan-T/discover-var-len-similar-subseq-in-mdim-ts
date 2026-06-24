import numpy as np


class BaseDtwDistance:
    """
    Base class for DTW distance calculation.
    """

    def calc(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the DTW distance of each .
        """
        raise NotImplementedError("Subclasses should implement this method!")

    def invert(self, x: np.ndarray) -> np.ndarray:
        """
        Invert the DTW distance of each .
        """
        raise NotImplementedError("Subclasses should implement this method!")




class AbsoluteDtwDistance(BaseDtwDistance):
    """
    Class for calculating absolute DTW distance.
    """
    def calc(self, x):
        """
        Calculate the absolute DTW distance.
        """
        return np.abs(x)

    def invert(self, x):
        """
        Invert the absolute DTW distance.
        """
        return np.abs(x)



class SquaredDtwDistance(BaseDtwDistance):
    """
    Class for calculating squared DTW distance.
    """
    def calc(self, x):
        """
        Calculate the squared DTW distance.
        """
        return np.square(x)

    def invert(self, x):
        """
        Invert the squared DTW distance.
        """
        return np.sqrt(x)


